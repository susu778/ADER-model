import torch
from torch import nn
from transformers import BertConfig, BertModel, BertPreTrainedModel

from ader import sampling, util


def get_token(hidden_states: torch.Tensor, input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
    """Return hidden states corresponding to a specific token, such as [CLS]."""
    hidden_size = hidden_states.shape[-1]
    flat_hidden = hidden_states.view(-1, hidden_size)
    flat_input = input_ids.contiguous().view(-1)
    return flat_hidden[flat_input == token_id, :]


class ADER(BertPreTrainedModel):
    """
    ADER model for joint entity and directed relation extraction.

    ADER uses span-based entity classification, decoupled head-tail
    cross-attention, and biaffine relation classification to model ordered
    biomedical entity pairs.
    """

    VERSION = "1.1"

    def __init__(
        self,
        config: BertConfig,
        cls_token=None,
        relation_types=None,
        entity_types=None,
        size_embedding=None,
        prop_drop=0.1,
        freeze_transformer=False,
        max_pairs=100,
    ):
        super().__init__(config)

        self._cls_token = cls_token if cls_token is not None else getattr(config, "cls_token", 101)
        self._relation_types = relation_types if relation_types is not None else getattr(config, "relation_types", 0)
        self._entity_types = entity_types if entity_types is not None else getattr(config, "entity_types", 0)
        self.size_embeddings_dim = size_embedding if size_embedding is not None else getattr(config, "size_embedding", 25)

        self.prop_drop = prop_drop
        self.freeze_transformer = freeze_transformer
        self._max_pairs = max_pairs

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(self.prop_drop)

        self.type_embedding_dim = 64
        self.entity_type_embeddings = nn.Embedding(self._entity_types, self.type_embedding_dim)
        self.size_embeddings = nn.Embedding(100, self.size_embeddings_dim)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + self.size_embeddings_dim, self._entity_types)

        self.ent_repr_dim = config.hidden_size + self.size_embeddings_dim + self.type_embedding_dim

        # Decoupled query projections for head and tail entity roles.
        self.head_q_proj = nn.Linear(self.ent_repr_dim, config.hidden_size)
        self.tail_q_proj = nn.Linear(self.ent_repr_dim, config.hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=self.prop_drop,
        )

        # Biaffine relation classifier for ordered entity pairs.
        self.biaffine_input_dim = self.ent_repr_dim + config.hidden_size
        self.biaffine_hidden_dim = 128
        self.head_mlp = nn.Sequential(
            nn.Linear(self.biaffine_input_dim, self.biaffine_hidden_dim),
            nn.ELU(),
            nn.Dropout(self.prop_drop),
        )
        self.tail_mlp = nn.Sequential(
            nn.Linear(self.biaffine_input_dim, self.biaffine_hidden_dim),
            nn.ELU(),
            nn.Dropout(self.prop_drop),
        )

        self.biaffine_U = nn.Parameter(
            torch.Tensor(self._relation_types, self.biaffine_hidden_dim, self.biaffine_hidden_dim)
        )
        self.biaffine_bias = nn.Parameter(torch.zeros(self._relation_types))

        self.init_weights()
        nn.init.xavier_uniform_(self.biaffine_U)

        if self.freeze_transformer:
            print("Freeze transformer weights")
            for param in self.bert.parameters():
                param.requires_grad = False

    def _truncate_inputs(self, encodings, context_masks, entity_masks):
        """Truncate sequences to the maximum BERT length."""
        if encodings.dim() == 2:
            encodings = encodings[:, :512]
        elif encodings.dim() == 3:
            encodings = encodings[:, :512, :]

        if context_masks.dim() == 2:
            context_masks = context_masks[:, :512]
        elif context_masks.dim() == 3:
            context_masks = context_masks[:, :512, :]

        if entity_masks.dim() == 3:
            entity_masks = entity_masks[:, :, :512]

        return encodings, context_masks, entity_masks

    def _forward_train(
        self,
        encodings: torch.Tensor,
        context_masks: torch.Tensor,
        entity_masks: torch.Tensor,
        entity_sizes: torch.Tensor,
        relations: torch.Tensor,
        rel_masks: torch.Tensor,
    ):
        encodings, context_masks, entity_masks = self._truncate_inputs(encodings, context_masks, entity_masks)

        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)["last_hidden_state"]
        batch_size = encodings.shape[0]

        safe_entity_sizes = torch.clamp(entity_sizes, min=0, max=99)
        size_embeddings = self.size_embeddings(safe_entity_sizes)
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)
        entity_type_indices = entity_clf.argmax(dim=-1)

        h_large = h.unsqueeze(1).expand(-1, max(min(relations.shape[1], self._max_pairs), 1), -1, -1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types], device=self.biaffine_U.device)

        for i in range(0, relations.shape[1], self._max_pairs):
            chunk_rel_logits = self._classify_relations(
                entity_spans_pool,
                size_embeddings,
                relations,
                rel_masks,
                h_large,
                i,
                entity_type_indices,
            )
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf

    def _forward_inference(
        self,
        encodings: torch.Tensor,
        context_masks: torch.Tensor,
        entity_masks: torch.Tensor,
        entity_sizes: torch.Tensor,
        entity_spans: torch.Tensor,
        entity_sample_masks: torch.Tensor,
    ):
        encodings, context_masks, entity_masks = self._truncate_inputs(encodings, context_masks, entity_masks)

        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)["last_hidden_state"]
        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        safe_entity_sizes = torch.clamp(entity_sizes, min=0, max=99)
        size_embeddings = self.size_embeddings(safe_entity_sizes)
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)
        entity_type_indices = entity_clf.argmax(dim=-1)

        relations, rel_masks, rel_sample_masks = self._filter_spans(
            entity_clf,
            entity_spans,
            entity_sample_masks,
            ctx_size,
        )
        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).expand(-1, max(min(relations.shape[1], self._max_pairs), 1), -1, -1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types], device=self.biaffine_U.device)

        for i in range(0, relations.shape[1], self._max_pairs):
            chunk_rel_logits = self._classify_relations(
                entity_spans_pool,
                size_embeddings,
                relations,
                rel_masks,
                h_large,
                i,
                entity_type_indices,
            )
            rel_clf[:, i:i + self._max_pairs, :] = torch.sigmoid(chunk_rel_logits)

        rel_clf = rel_clf * rel_sample_masks
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        mask = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = mask + h.unsqueeze(1).expand(-1, entity_masks.shape[1], -1, -1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]
        entity_ctx = get_token(h, encodings, self._cls_token)

        entity_repr = torch.cat(
            [
                entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                entity_spans_pool,
                size_embeddings,
            ],
            dim=2,
        )
        entity_repr = self.dropout(entity_repr)
        entity_clf = self.entity_classifier(entity_repr)
        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start, entity_type_indices):
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        entity_pairs = util.batch_index(entity_spans, relations)
        head_entity = entity_pairs[:, :, 0, :]
        tail_entity = entity_pairs[:, :, 1, :]

        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        head_size = size_pair_embeddings[:, :, 0, :]
        tail_size = size_pair_embeddings[:, :, 1, :]

        type_pairs = util.batch_index(entity_type_indices, relations)
        type_emb_pairs = self.entity_type_embeddings(type_pairs)
        head_type = type_emb_pairs[:, :, 0, :]
        tail_type = type_emb_pairs[:, :, 1, :]

        head_repr = torch.cat([head_entity, head_size, head_type], dim=-1)
        tail_repr = torch.cat([tail_entity, tail_size, tail_type], dim=-1)

        # Head and tail entities independently query the same sentence context.
        batch_size, pair_count, seq_len, hidden_size = h.shape
        head_query = self.head_q_proj(head_repr).view(batch_size * pair_count, 1, hidden_size)
        tail_query = self.tail_q_proj(tail_repr).view(batch_size * pair_count, 1, hidden_size)
        combined_query = torch.cat([head_query, tail_query], dim=1)

        kv_flat = h.contiguous().view(batch_size * pair_count, seq_len, hidden_size)
        key_padding_mask = (rel_masks == 0).view(batch_size * pair_count, seq_len)
        all_padded = key_padding_mask.all(dim=-1)
        key_padding_mask[all_padded, 0] = False

        combined_ctx, _ = self.cross_attn(
            query=combined_query,
            key=kv_flat,
            value=kv_flat,
            key_padding_mask=key_padding_mask,
        )

        head_ctx = combined_ctx[:, 0, :].view(batch_size, pair_count, hidden_size)
        tail_ctx = combined_ctx[:, 1, :].view(batch_size, pair_count, hidden_size)
        head_ctx[all_padded.view(batch_size, pair_count)] = 0.0
        tail_ctx[all_padded.view(batch_size, pair_count)] = 0.0

        head_features = torch.cat([head_repr, head_ctx], dim=-1)
        tail_features = torch.cat([tail_repr, tail_ctx], dim=-1)

        head_mapped = self.head_mlp(head_features)
        tail_mapped = self.tail_mlp(tail_features)

        rel_logits = torch.einsum("bpd,rde,bpe->bpr", head_mapped, self.biaffine_U, tail_mapped)
        rel_logits = rel_logits + self.biaffine_bias

        return rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        device = self.biaffine_U.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, inference=False, **kwargs):
        if inference:
            return self._forward_inference(*args, **kwargs)
        return self._forward_train(*args, **kwargs)


_MODELS = {
    "ader": ADER,
}


def get_model(name):
    if name not in _MODELS:
        raise ValueError(f"Unknown model type: {name}. Available models: {list(_MODELS.keys())}")
    return _MODELS[name]
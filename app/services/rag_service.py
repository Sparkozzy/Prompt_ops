import hashlib
import math
import re
from typing import Dict, Any, List, Optional, Set
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload
from app.models import RagParentPrompt, RagChildChunk, get_utc_now
from app.services.llm_service import get_openai_client


# Tags modulares prioritárias da arquitetura MindFlow
CANONICAL_XML_TAGS = [
    "GlobalSettings",
    "variaveis",
    "InformacoesVariaveis",
    "persona",
    "Persona",
    "Identidade",
    "Identity",
    "regras_comunicacao",
    "ferramentas_MCP",
    "fluxo_de_uso",
    "few_shot_declarado",
    "MANDATORY_TOOL_CALL",
    "Mandatory",
    "FailSafe",
    "ContextoDeOrigem",
]


def parse_prompt_into_parent_child_chunks(prompt_text: str) -> List[Dict[str, Any]]:
    """
    Quebra um prompt corporativo em Child Chunks semânticos preservando blocos XML ou seções Markdown.
    Caso o prompt seja um bloco único sem tags ou headers, divide em janelas lógicas.
    """
    if not prompt_text or not prompt_text.strip():
        return []

    text = prompt_text.strip()
    chunks: List[Dict[str, Any]] = []

    # 1. Extrai blocos XML explícitos (<Tag>...</Tag>)
    # Ignora wrappers raiz gigantes como <SystemPrompt> ou <Prompt> para extrair os filhos reais
    xml_pattern = re.compile(
        r"<([a-zA-Z0-9_-]+)(?:\s[^>]*)?>(.*?)</\1>",
        re.DOTALL
    )
    root_wrappers = {"SystemPrompt", "Prompt", "System"}

    # Se o texto estiver envolto em <SystemPrompt>...</SystemPrompt>, analisa o interior também
    search_targets = [text]
    for wrapper in root_wrappers:
        w_match = re.findall(rf"<{wrapper}(?:\s[^>]*)?>(.*?)</{wrapper}>", text, re.DOTALL | re.IGNORECASE)
        for inner in w_match:
            if inner and inner not in search_targets:
                search_targets.append(inner)

    seen_hashes: Set[str] = set()
    idx = 0

    for target in search_targets:
        for match in xml_pattern.finditer(target):
            tag_name = match.group(1)
            if tag_name in root_wrappers:
                continue
            full_block = match.group(0).strip()
            if len(full_block) < 25:
                continue
            h = hashlib.md5(full_block.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            chunks.append({
                "section_tag": tag_name,
                "chunk_index": idx,
                "chunk_text": full_block[:6000],
            })
            idx += 1

    # 2. Extrai seções Markdown (### / ## / #) que não estejam duplicadas
    md_sections = re.split(r"(?m)^(#{1,3}\s+.+)$", text)
    if len(md_sections) > 1:
        for i in range(1, len(md_sections), 2):
            header = md_sections[i].strip()
            body = md_sections[i + 1].strip() if (i + 1) < len(md_sections) else ""
            full_sec = f"{header}\n{body}".strip()
            if len(full_sec) < 40:
                continue
            h = hashlib.md5(full_sec.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            clean_title = re.sub(r"^#{1,3}\s+", "", header).strip()[:80]
            chunks.append({
                "section_tag": f"md:{clean_title}",
                "chunk_index": idx,
                "chunk_text": full_sec[:6000],
            })
            idx += 1

    # 3. Fallback caso o prompt não tenha tags XML filhas nem headers Markdown
    if not chunks:
        step = 3000
        for start in range(0, len(text), step):
            piece = text[start:start + step].strip()
            if piece:
                chunks.append({
                    "section_tag": "general_block",
                    "chunk_index": idx,
                    "chunk_text": piece,
                })
                idx += 1

    return chunks


async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Gera embeddings em lote usando text-embedding-3-small da OpenAI."""
    if not texts:
        return []
    try:
        client = get_openai_client()
        cleaned = [t.replace("\n", " ")[:8000] for t in texts]
        resp = await client.embeddings.create(
            model="text-embedding-3-small",
            input=cleaned
        )
        return [item.embedding for item in resp.data]
    except Exception as e:
        print(f"[RAG Embedding Warning] Falha ao gerar embeddings OpenAI, usando fallback vetorial: {e}")
        return [[] for _ in texts]


def _cosine_similarity(vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_overlap_score(query: str, text: str, section_tag: str) -> float:
    q_tokens = set(re.findall(r"[a-zA-ZÀ-ÿ0-9_]{3,}", query.lower()))
    if not q_tokens:
        return 0.0
    target = f"{section_tag} {text}".lower()
    matches = sum(1 for tok in q_tokens if tok in target)
    return matches / len(q_tokens)


async def upsert_prompt_in_rag(
    session: AsyncSession,
    prompt_id: int,
    client_id: str,
    client_name: Optional[str],
    prompt_name: Optional[str],
    channel: str,
    prompt_text: str,
    version_label: str = "prod",
    eval_score: Optional[float] = None,
    metadata_json: Optional[Dict[str, Any]] = None,
) -> RagParentPrompt:
    """
    Substitui atomicamente o Parent anterior para (prompt_id, channel).
    Graças ao ON DELETE CASCADE e cascade="all, delete-orphan", todos os Child Chunks
    da versão obsoleta são excluídos automaticamente antes da inserção dos novos chunks.
    """
    ch = (channel or "whatsapp").lower()
    content_hash = hashlib.sha256((prompt_text or "").encode("utf-8")).hexdigest()

    # 1. Busca e remove qualquer Parent existente para (prompt_id, channel)
    stmt_existing = select(RagParentPrompt).options(selectinload(RagParentPrompt.children)).where(
        RagParentPrompt.prompt_id == int(prompt_id),
        RagParentPrompt.channel == ch
    )
    res_existing = await session.execute(stmt_existing)
    existing_parents = res_existing.scalars().all()
    for old_parent in existing_parents:
        await session.delete(old_parent)
    if existing_parents:
        await session.flush()

    # 2. Quebra em Child Chunks e gera embeddings em batch
    parsed_chunks = parse_prompt_into_parent_child_chunks(prompt_text)
    embed_inputs = [
        f"Cliente: {client_name or ''} | Canal: {ch} | Bloco: {c['section_tag']}\n{c['chunk_text']}"
        for c in parsed_chunks
    ]
    embeddings = await generate_embeddings_batch(embed_inputs)

    # 3. Cria novo Parent e seus novos Child Chunks
    new_parent = RagParentPrompt(
        prompt_id=int(prompt_id),
        client_id=str(client_id),
        client_name=client_name or "Cliente",
        prompt_name=prompt_name or "Prompt",
        channel=ch,
        version_label=version_label,
        eval_score=eval_score,
        content_hash=content_hash,
        full_prompt_text=prompt_text,
        metadata_json=metadata_json or {"active_client": True},
        created_at=get_utc_now(),
        updated_at=get_utc_now(),
    )
    session.add(new_parent)
    await session.flush()

    for idx, c in enumerate(parsed_chunks):
        emb = embeddings[idx] if idx < len(embeddings) else []
        child = RagChildChunk(
            parent_id=new_parent.id,
            prompt_id=int(prompt_id),
            channel=ch,
            section_tag=c["section_tag"],
            chunk_index=c["chunk_index"],
            chunk_text=c["chunk_text"],
            embedding=emb,
            created_at=get_utc_now(),
        )
        session.add(child)

    await session.commit()
    await session.refresh(new_parent)
    return new_parent


async def search_rag_chunks(
    session: AsyncSession,
    query: str,
    section_tag: Optional[str] = None,
    channel: Optional[str] = None,
    exclude_prompt_id: Optional[int] = None,
    include_parent_text: bool = False,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Executa busca híbrida (Similaridade Cosseno via Embeddings + Overlap Lexical)
    nos Child Chunks indexados, retornando também os metadados do Parent.
    """
    stmt = select(RagChildChunk).options(selectinload(RagChildChunk.parent))
    if section_tag:
        stmt = stmt.where(RagChildChunk.section_tag.ilike(f"%{section_tag}%"))
    if channel:
        stmt = stmt.where(RagChildChunk.channel == channel.lower())
    if exclude_prompt_id is not None:
        stmt = stmt.where(RagChildChunk.prompt_id != int(exclude_prompt_id))

    res = await session.execute(stmt)
    all_chunks = res.scalars().all()
    if not all_chunks:
        return []

    query_vecs = await generate_embeddings_batch([query])
    q_vec = query_vecs[0] if query_vecs else []

    scored: List[Dict[str, Any]] = []
    for c in all_chunks:
        cos_sim = _cosine_similarity(q_vec, c.embedding if isinstance(c.embedding, list) else None)
        lex_sim = _lexical_overlap_score(query, c.chunk_text, c.section_tag)
        # Combina similaridade vetorial (peso 0.75) + lexical (peso 0.25)
        hybrid_score = (cos_sim * 0.75) + (lex_sim * 0.25) if q_vec else lex_sim
        parent = c.parent
        parent_info = {
            "prompt_id": c.prompt_id,
            "client_id": parent.client_id if parent else "",
            "client_name": parent.client_name if parent else "",
            "prompt_name": parent.prompt_name if parent else "",
            "channel": c.channel,
            "version_label": parent.version_label if parent else "",
            "eval_score": parent.eval_score if parent else None,
        }
        if include_parent_text and parent:
            parent_info["full_prompt_text"] = parent.full_prompt_text[:6000]

        scored.append({
            "chunk_id": str(c.id),
            "section_tag": c.section_tag,
            "chunk_index": c.chunk_index,
            "chunk_text": c.chunk_text,
            "similarity_score": round(hybrid_score, 4),
            "parent": parent_info,
        })

    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored[:top_k]


def get_allowed_seed_prompt_ids(
    client_configs: List[Dict[str, Any]],
    prompts_rows: List[Dict[str, Any]]
) -> Set[int]:
    """
    Determina estritamente os IDs de prompts autorizados para o Seed do RAG:
    1. Prompts vinculados a clientes ativos em client_configurations (por prompt_id ou nome de cliente ativo).
    2. Prompts de Treinamento de SDR (ex: IDs 32 e 67 ou cujo nome contenha 'treinamento sdr').
    """
    allowed: Set[int] = set()
    active_client_names: Set[str] = set()

    for cfg in client_configs:
        pid = cfg.get("prompt_id")
        if pid is not None:
            try:
                allowed.add(int(pid))
            except (ValueError, TypeError):
                pass
        cname = (cfg.get("client_name") or "").strip().lower()
        if cname:
            active_client_names.add(cname)

    for p in prompts_rows:
        pid = int(p["id"])
        p_client = (p.get("Nome do cliente") or "").strip().lower()
        p_name = (p.get("Pormpt_Name") or "").strip().lower()
        combined = f"{p_client} {p_name}"

        # Inclui Treinamento de SDR
        if "treinamento sdr" in combined or pid in {32, 67}:
            allowed.add(pid)
            continue

        # Inclui se o nome do cliente bate exatamente com um cliente ativo de client_configurations (ex: Versa, ATS)
        if p_client and p_client in active_client_names:
            allowed.add(pid)

    return allowed


async def seed_active_and_sdr_prompts_to_rag(session: AsyncSession) -> Dict[str, Any]:
    """
    Executa o seed no RAG Parent-Child consultando apenas os prompts de clientes ativos
    (client_configurations) + Treinamento de SDR (32 e 67).
    """
    from app.database import get_production_supabase
    supabase = get_production_supabase()

    cfg_res = supabase.table("client_configurations").select("client_id, client_name, prompt_id").execute()
    configs = cfg_res.data or []

    prompts_res = supabase.table("Prompts").select(
        'id, "Nome do cliente", "Pormpt_Name", "Prompt_Text", "Ligação/txt", "prompt_fup", "Prompt Insta"'
    ).execute()
    all_prompts = prompts_res.data or []

    allowed_ids = get_allowed_seed_prompt_ids(configs, all_prompts)
    prompt_to_client_id = {
        int(c["prompt_id"]): str(c["client_id"])
        for c in configs if c.get("prompt_id") is not None
    }

    indexed_parents = 0
    indexed_prompt_ids = []

    channel_columns = [
        ("whatsapp", "Prompt_Text"),
        ("ligacao", "Ligação/txt"),
        ("fup", "prompt_fup"),
        ("instagram", "Prompt Insta"),
    ]

    for p in all_prompts:
        pid = int(p["id"])
        if pid not in allowed_ids:
            continue

        client_name = (p.get("Nome do cliente") or "Cliente").strip()
        prompt_name = (p.get("Pormpt_Name") or client_name).strip()
        cid = prompt_to_client_id.get(pid, str(pid))

        for ch_key, col_name in channel_columns:
            raw_text = p.get(col_name)
            if raw_text and str(raw_text).strip():
                await upsert_prompt_in_rag(
                    session=session,
                    prompt_id=pid,
                    client_id=cid,
                    client_name=client_name,
                    prompt_name=prompt_name,
                    channel=ch_key,
                    prompt_text=str(raw_text).strip(),
                    version_label=f"prod-{ch_key}",
                    eval_score=1.0,
                    metadata_json={
                        "is_sdr_training": ("sdr" in client_name.lower() or pid in {32, 67}),
                        "active_client": True,
                    },
                )
                indexed_parents += 1
        indexed_prompt_ids.append(pid)

    return {
        "success": True,
        "indexed_prompt_ids": sorted(indexed_prompt_ids),
        "total_parent_channels_indexed": indexed_parents,
    }

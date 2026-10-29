import pytest
from app.services.llm_service import get_default_datetime_variables, render_prompt_template

def test_get_default_datetime_variables():
    vars_dict = get_default_datetime_variables()
    required_keys = [
        "data_atual", "hora_atual", "horario_atual", "data_hora_atual",
        "data_atual_iso", "dia_da_semana", "dia_semana", "now", "$now",
        "current_date", "current_time", "current_datetime", "data", "hora"
    ]
    for key in required_keys:
        assert key in vars_dict, f"Chave obrigatória {key} ausente no dicionário de datetime."
        assert vars_dict[key], f"Valor da chave {key} está vazio."

def test_render_prompt_template_substitutes_placeholders():
    template = "Olá! Hoje é {data_atual} às {hora_atual}. Padrão ISO: {data_atual_iso}. Padrão n8n: {{$now}}."
    rendered = render_prompt_template(template)
    
    assert "{data_atual}" not in rendered
    assert "{hora_atual}" not in rendered
    assert "{data_atual_iso}" not in rendered
    assert "{{$now}}" not in rendered
    assert "/" in rendered  # DD/MM/YYYY

def test_render_prompt_template_injects_temporal_context_when_missing():
    template = "<SystemPrompt>\n<persona>Você é a assistente Kravi.</persona>\n</SystemPrompt>"
    rendered = render_prompt_template(template)
    
    assert "<contexto_temporal>" in rendered
    assert "</contexto_temporal>" in rendered
    assert "Data e horário atual (horário de Brasília UTC-3):" in rendered

def test_render_prompt_template_respects_custom_variables_override():
    template = "Atendimento teste: {data_atual_iso} no canal {canal}"
    custom_vars = {"data_atual_iso": "2026-10-19T12:05:00-03:00", "canal": "whatsapp"}
    rendered = render_prompt_template(template, variables=custom_vars)
    
    assert "2026-10-19T12:05:00-03:00" in rendered
    assert "whatsapp" in rendered

# Integridade de deletes

Leituras do AI-Lake são permissivas por padrão para preservar compatibilidade:
se uma deletion vector ou um equality delete não puder ser carregado, a
consulta registra o problema e continua. Isso pode expor temporariamente uma
linha potencialmente excluída.

Para workloads que exigem fail-closed, use `strict_deletes`:

```rust
let config = ailake_query::SearchConfig {
    strict_deletes: true,
    ..Default::default()
};
```

Com `strict_deletes = true`, falham imediatamente:

- erros ao ler ou desserializar deletion vectors;
- erros ao listar equality-delete manifests;
- erros ao ler ou interpretar arquivos Avro de equality deletes.

O modo também é propagado por buscas multimodais. Para busca textual, a API
equivalente é `search_text_with_options(..., strict_deletes)`. No CLI:

```bash
ailake search tabela --query "0.1,0.2,0.3" --strict-deletes
ailake search tabela --text "termo" --strict-deletes
ailake serve tabela --strict-deletes
ailake read-changes tabela --strict-deletes
```

O CDC (`ChangeReaderConfig`) aplica a mesma política aos deletion vectors. O
modo permissivo continua disponível explicitamente com `false` e é o default
para bindings Python, JNI e chamadas existentes.

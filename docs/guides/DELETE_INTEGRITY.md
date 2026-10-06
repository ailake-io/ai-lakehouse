# Integridade de deletes

As buscas Rust e os comandos `ailake search` e `ailake serve` falham de forma
fechada por padrão: se uma deletion vector ou um equality delete não puder ser
carregado, a consulta falha em vez de devolver uma linha potencialmente
excluída. `SearchConfig::default()` e a função Rust `search_text()` usam esse
comportamento.

O modo pode ser definido explicitamente na API Rust:

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

O modo também é propagado por buscas multimodais. Para busca textual, use
`search_text_with_options(..., strict_deletes)`. No CLI:

```bash
ailake search tabela --query "0.1,0.2,0.3"
ailake search tabela --text "termo"
ailake serve tabela
ailake read-changes tabela --strict-deletes
```

No CLI, `search` e `serve` são estritos por padrão. `--allow-stale-deletes` (ou
`AILAKE_ALLOW_STALE_DELETES=1` no servidor) volta explicitamente ao modo
permissivo. `--strict-deletes` permanece aceito por compatibilidade.

## Limites atuais por integração

- Os bindings Python e JNI ainda constroem `SearchConfig` com
  `strict_deletes: false` e não expõem uma opção para alterá-lo. Buscas por essas
  integrações podem retornar linhas excluídas se os arquivos de delete não
  puderem ser lidos. A exposição de uma opção estrita nesses bindings continua
  pendente.
- `SearchSession::search_query()` usa índices pré-carregados e não aplica
  deletion vectors nem equality deletes. Não use essa sessão para resultados que
  precisem refletir deletes atuais; prefira o caminho normal de `search()`.
- `ailake read-changes` é uma API de CDC separada. Seu
  `ChangeReaderConfig::strict_deletes` continua independente e permissivo por
  padrão; passe `--strict-deletes` no CLI quando necessário.

Esses limites são relevantes para correção dos resultados, não apenas para
disponibilidade. Planeje corrigir a configuração nos bindings e a visibilidade
de deletes em `SearchSession` antes de usá-los em fluxos que exigem integridade
de estado atual.

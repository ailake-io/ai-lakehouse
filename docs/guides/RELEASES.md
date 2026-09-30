# Releases e versionamento

O AI-Lake usa uma versão única para crates Rust, wheel Python, provider Airflow,
plugins JVM, C-ABI e extensão DuckDB. A versão publicada mais recente é sempre
a última tag SemVer; a próxima versão é derivada automaticamente pela Action
`Release`.

## Fluxo automático

1. O merge de `develop` para `main` dispara `.github/workflows/release.yml`.
2. A Action espera todos os checks obrigatórios terminarem com sucesso.
3. O patch é incrementado a partir da última tag (`vX.Y.Z` → próximo patch), ou
   `version_override` é usado em uma execução manual.
4. Crate manifests, artefatos JVM, CMake, providers e exemplos documentais são
   atualizados para a nova versão.
5. A seção `[Unreleased]` do `CHANGELOG.md` vira a seção da versão publicada,
   com a data UTC do release, e uma nova seção `[Unreleased]` é deixada no topo.
6. A Action cria a tag e o GitHub Release e publica crates, JARs, provider
   Airflow e wheels Python na sequência.

Não edite manualmente a versão dos artefatos para um release normal. Durante o
desenvolvimento, registre mudanças visíveis ao usuário em `[Unreleased]`:

```markdown
## [Unreleased]

### Added

- Descreva a capacidade e indique as superfícies afetadas.
```

## Checklist antes do merge em `main`

- `[Unreleased]` contém as mudanças da janela de release.
- Exemplos de instalação usam a versão atualmente publicada (`TAG` e
  `JAR_VERSION`); a Action os atualiza na promoção da tag.
- CI, CI Safety, Go, C++, performance e Compat Heavy estão verdes.
- Mudanças de comportamento operacional apontam para as guias de cache,
  coordenação distribuída, secrets, rate limiting e performance.

Para reprocessar apenas um artefato depois de uma falha, use os workflows de
fallback documentados em [`docs/contributing/TESTING.md`](../contributing/TESTING.md).

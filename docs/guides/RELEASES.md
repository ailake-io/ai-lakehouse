# Releases e versionamento

O core Rust usa a tag `vX.Y.Z`. O SDK Python, cada plugin e a integração Kof
possuem tags e ciclos independentes. A C-ABI mantém seu próprio número de
contrato para compatibilidade binária. Cada Action incrementa a versão a partir
da última tag do componente correspondente.

## Fluxo automático

1. O merge de `develop` para `main` dispara `.github/workflows/release.yml`.
2. A Action espera todos os checks obrigatórios terminarem com sucesso.
3. O patch é incrementado a partir da última tag (`vX.Y.Z` → próximo patch), ou
   `version_override` é usado em uma execução manual.
4. Os manifests e exemplos documentais do core são atualizados para a nova versão.
5. A seção `[Unreleased]` do `CHANGELOG.md` vira a seção da versão publicada,
   com a data UTC do release, e uma nova seção `[Unreleased]` é deixada no topo.
6. A Action cria a tag e o GitHub Release do core e publica crates e a biblioteca
   JNI. Workflows separados publicam os SDKs e integrações.

Não edite manualmente a versão dos artefatos para um release normal. Durante o
desenvolvimento, registre mudanças visíveis ao usuário em `[Unreleased]`:

```markdown
## [Unreleased]

### Added

- Descreva a capacidade e indique as superfícies afetadas.
```

## Checklist antes do merge em `main`

- `[Unreleased]` contém as mudanças da janela de release.
- Exemplos do core usam `vX.Y.Z`; SDKs e plugins usam tags próprias pelo
  workflow nomeado correspondente. Python usa `Release Python`; os plugins
  usam a implementação compartilhada `release-plugin.yml`; os prefixos ficam
  no registry `scripts/plugins.json`.
- CI, CI Safety, Go, C++, performance e Compat Heavy estão verdes.
- Mudanças de comportamento operacional apontam para as guias de cache,
  coordenação distribuída, secrets, rate limiting e performance.

Para reprocessar apenas um artefato depois de uma falha, use os workflows de
fallback documentados em [`docs/contributing/TESTING.md`](../contributing/TESTING.md).

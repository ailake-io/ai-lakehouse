# Releases e versionamento

O core Rust usa a tag `vX.Y.Z`. O SDK Python usa a mesma versão do core, com
tag `python-vX.Y.Z`. O release do core chama automaticamente a Action Python
para publicar wheels e sdist com o mesmo número. Cada plugin e a
integração Kof mantêm tags e ciclos independentes. A C-ABI mantém seu próprio
número de contrato para compatibilidade binária.

## Publicação por tag

Merges e pushes em `main` executam CI, mas não publicam versões. A publicação
do core começa apenas ao enviar uma tag estável `vX.Y.Z`.

1. Prepare as versões e as notas em um PR. Durante o desenvolvimento, continue
   registrando mudanças visíveis ao usuário em `[Unreleased]`.
2. Para preparar uma versão futura, por exemplo `0.1.25` a partir de `0.1.24`,
   execute na branch do PR:

   ```bash
   python3 scripts/plugins.py bump-core-cargo --from-version 0.1.24 --version 0.1.25
   python3 scripts/plugins.py update --policy core --version 0.1.25
   cargo metadata --format-version 1 >/dev/null
   cargo metadata --format-version 1 --manifest-path fuzz/Cargo.toml >/dev/null
   ```

   Atualize também os exemplos do core na documentação e promova as notas de
   `[Unreleased]` para `## [0.1.25] — AAAA-MM-DD`, deixando uma nova seção
   `[Unreleased]` no topo. Revise o diff e inclua os dois lockfiles no PR.
   O Python acompanha o core; os demais plugins preservam versões independentes.
3. Faça merge do PR em `main` e espere a CI desse commit passar. Não use
   `[skip ci]` no commit de preparação ou na mensagem do merge: a publicação
   exige uma execução de CI bem-sucedida na `main` para o commit da tag.
4. Atualize a cópia local da `main`, valide e envie a tag no commit aprovado:

   ```bash
   git fetch origin main
   git switch --detach origin/main
   python3 scripts/release_tag.py v0.1.25
   git tag -a v0.1.25 -m "Release v0.1.25"
   git push origin v0.1.25
   ```

5. `release.yml` valida o formato da tag, confirma que o commit pertence à
   história da `main`, verifica as versões dos manifests e lockfiles e exige
   uma seção datada no changelog. Aguarda a CI e os demais checks por até
   90 minutos, então cria o GitHub Release e publica crates e a biblioteca JNI.
6. O job `sync-develop` continua levando o commit publicado para `develop`.
   Após a publicação dos crates e JNI, o workflow chama `Release Python`
   automaticamente com a mesma versão. O Python usa o commit da tag do core.
   Os demais plugins continuam usando seus workflows próprios.

O workflow publica o código do commit da tag sem incrementar versões, criar
commits em `main` ou mover a tag. Uma tag incompatível com as versões preparadas
falha antes da publicação. Para repetir uma publicação após falha, use **Re-run
failed jobs** na execução da tag; não apague nem mova uma tag publicada.

## Verificações antes da tag

- A versão do core e os dois lockfiles correspondem à tag.
- O changelog tem a seção datada da versão e uma nova seção `[Unreleased]`.
- Exemplos e referências do core usam a versão preparada.
- CI, CI Safety, Go, C++, performance e Compat Heavy estão verdes.
- O PR foi integrado com merge commit, conforme a regra da branch.
- Merges não publicam artefatos. Ao enviar a tag do core, a publicação Python
  correspondente faz parte do mesmo workflow. Se ela falhar, reexecute os jobs
  falhos; não crie outra versão do core para tentar novamente.
- Os demais plugins usam `release-plugin.yml`, com versões e exemplos preparados
  em PRs e prefixos definidos em `scripts/plugins.json`. Veja o
  [catálogo de versões e artefatos](PLUGIN_RELEASES.md).

`Cargo.lock` e `fuzz/Cargo.lock` são versionados e o job `lockfiles` do CI falha
se algum estiver defasado. Ao mudar um `Cargo.toml`, commite o lock junto.
O workflow `cargo-update.yml` abre atualizações semanais contra `develop`.
Os alertas do Dependabot continuam ativos.

Para reprocessar apenas um artefato depois de uma falha, use os workflows de
fallback documentados em [`docs/contributing/TESTING.md`](../contributing/TESTING.md).


## Recuperação de uma publicação Python

`Release Python` e `Republish Python SDK` continuam disponíveis para recuperação
manual a partir de `main`. Informe a versão do core já publicado ou deixe vazia
para usar a versão atual. A Action exige o release do core existente, usa sua
tag como fonte e recusa versões que não correspondam ao core dessa tag.
Se a tag Python já existe, o commit original é preservado.

O manifest Python e os exemplos `ailake==X.Y.Z` são parte da política `core`
do registry. O comando `plugins.py update --policy core` os atualiza juntos;
a CI detecta divergências antes de uma nova publicação.

A publicação só está concluída quando o workflow inteiro, incluindo o Python
e a verificação dos artefatos publicados, terminar com sucesso.

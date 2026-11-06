# Estudo: plugin Trino para SPI 475+

> **Status**: estudo, não implementado (2026-11-06). Motivação: 15 alertas do
> Dependabot sobre o Jackson 2.15.2 empacotado no plugin, dispensados como
> `not_used` até que este port seja feito.

## 1. Problema

O plugin (`trino-plugin/`) tem dois problemas com a mesma causa: ele é compilado
contra o SPI do Trino **430** (outubro de 2023).

1. **Jackson vulnerável no JAR.** O plugin empacota `jackson-databind` e
   `jackson-core` 2.15.2, com 15 advisories (12 + 3). A versão corrigida mais
   próxima é a 2.18.12.
2. **O plugin não funciona em Trino 460 ou mais novo.** Na inicialização do
   catálogo, falha com `ConnectorMetadata getTableHandle() is not implemented`
   (registrado em [`JVM_INTEGRATION.md`](../guides/JVM_INTEGRATION.md) §5A). Por
   isso a documentação manda fixar o servidor na 430.

### Por que não dá para só subir o Jackson

O classloader de plugins do Trino carrega o pacote
`com.fasterxml.jackson.annotation` **do servidor**, por cima da cópia que vem no
plugin: é o mesmo pacote usado pelos `@JsonProperty`/`@JsonCreator` dos handles,
que o próprio Trino serializa entre coordinator e worker. O `jackson-databind`
empacotado, porém, vem do plugin. Um `databind` mais novo que as anotações do
servidor pode falhar em tempo de execução: o 2.16+ referencia valores de enum (por
exemplo, `JsonTypeInfo.Id.SIMPLE_NAME`) que não existem nas anotações 2.15 do
Trino 430.

Por isso o Jackson do plugin precisa ser, no máximo, da mesma linha das anotações
do servidor mais antigo suportado. Anotações mais novas que o `databind` não são
problema: o pacote de anotações só cresce.

## 2. Opções

| Opção | Fecha os CVEs | Corrige Trino 460+ | Mantém Trino 430 | Custo |
|---|---|---|---|---|
| **A. Port para SPI 475** | sim | sim | não | médio: 8 erros de compilação + validação ao vivo |
| B. Trocar Jackson `databind` por outra lib JSON no código do plugin | sim | não | sim | médio: cerca de 50 pontos de uso |
| C. Relocar (shade) o Jackson | não | não | sim | — |
| D. Manter como está (atual) | não, dispensados | não | sim | zero |

- **C não funciona.** Relocar o pacote de anotações quebra a serialização dos
  handles, porque o Trino deixa de enxergar os `@JsonProperty`. Relocar só
  `databind`/`core` mantém a dependência das anotações do servidor, que é o
  problema original.
- **B** resolve só os CVEs. O plugin continua quebrado em qualquer Trino da
  460 em diante. Os pontos de uso hoje são 18 `readValue`, 9
  `readTree`, 20 `writeValueAsString` e 5 de `ObjectNode`/`ArrayNode`. Os 72
  `@JsonProperty` dos handles ficam: eles são lidos pelo Jackson do servidor, não
  pelo do plugin.
- **Recomendação: A.** Resolve os dois problemas. O custo real é abandonar o
  Trino abaixo da 475. Hoje o plugin só foi verificado na 430 e quebra em
  qualquer versão da 460 em diante.

## 3. Versão alvo: 475

Levantado dos POMs e dos sources do `trino-spi` no Maven Central.

| Trino | Jackson (anotações) | Java do servidor | `addColumn` 3 args | `addColumn` + `ColumnPosition` | `shutdown()` abstrato |
|---|---|---|---|---|---|
| 430 | 2.15.2 | 17 | sim | não | não |
| 460 | 2.18.0 | 22 | sim | não | não |
| 470 | 2.18.2 | 23 | sim | sim | não |
| **475** | **2.18.3** | **23** | **não** | **sim** | **não** |
| 476 | 2.19.0 | 24 | não | sim | não |
| 478 | 2.20.0 | 24 | não | sim | sim |
| 480 | 2.21.2 | 25 | não | sim | sim |

- **475 é a menor versão em que um único JAR funciona até a 480.** O
  `addColumn(session, handle, column)` de 3 argumentos sumiu na 475. Na 470 ainda
  existem as duas formas. Um build contra a 470 implementaria a forma antiga e
  quebraria `ALTER TABLE ADD COLUMN` da 475 em diante.
- `shutdown()` virou abstrato na 478. Implementá-lo também funciona na 475, onde
  já existe com corpo padrão.
- **O Jackson do plugin pode subir para 2.18.12**, mesma linha das anotações da
  475 (2.18.3). Servidores 476+ trazem anotações 2.19 a 2.21, que são compatíveis.

## 4. Mudanças de código

Medido compilando cópias do plugin contra `trino-spi` 460 e 480 (Kotlin 1.9.23,
JDK 21, Gradle 8.7). Contra a 430 o mesmo build compila limpo.

| # | Arquivo | Erro | Correção |
|---|---|---|---|
| 1 | `VectorScanMetadata.kt:147` | `getTableHandle` 2-args não existe mais | implementar `getTableHandle(session, name, startVersion, endVersion)`. Com versão presente (time travel), lançar `NOT_SUPPORTED`. Na 475 o corpo padrão lança `getTableHandle() is not implemented`: é exatamente o erro que a 460 já dá. |
| 2 | `VectorScanMetadata.kt:219` | `finishInsert` 4-args não existe mais | novo parâmetro `sourceTableHandles: List<ConnectorTableHandle>` (ignorado). O padrão da 475 lança erro, então hoje o `INSERT` também quebraria. |
| 3 | `VectorScanMetadata.kt:311` | `addColumn` 3-args (só 475+) | `addColumn(session, handle, column, position: ColumnPosition)`. `ColumnPosition` é uma *sealed interface* com `First`, `After` e `Last`: aceitar `Last` e lançar `NOT_SUPPORTED` para os outros, como faz o padrão do SPI. |
| 4 | `VectorScanMetadata.kt:261` | construtor de 3 args de `ConstraintApplicationResult` removido | `ConstraintApplicationResult(newHandle, TupleDomain.all(), Constant.TRUE, false)`; `Constant.TRUE` existe na 430 e na 475. |
| 5 | `VectorScanHandles.kt:101` | `ConnectorSplit.getInfo()` removido | apagar o override |
| 6 | `VectorScanHandles.kt:119` | idem | apagar o override |
| 7 | `AilakePageSink.kt:66` | `Block.getObject(pos, Block::class.java)` removido | `ArrayType(DOUBLE).getObject(block, pos)`, que existe nas duas versões e já trata blocos de dicionário/RLE via `getUnderlyingValueBlock()` |
| 8 | `VectorScanConnector.kt:16` | `Connector.shutdown()` abstrato (478+) | implementar `override fun shutdown() {}` |

### Build e toolchain

- **Não é preciso trocar o Kotlin.** O Kotlin 1.9.23 compilou contra os SPIs 460
  (Java 22) e 480 (Java 25) sem erro de versão de bytecode; todos os erros foram
  de API. Isso evita mexer no catálogo `gradle/ailake-plugins.versions.toml`, que
  é compartilhado com os plugins Spark e Flink.
- O bytecode pode continuar Java 17: o servidor roda Java 23+, que executa
  bytecode 17. O spike usou `jvmToolchain(21)` por falta de JDK 17 local; falta
  confirmar o build com `jvmToolchain(17)`.
- `build.gradle.kts`: `trinoVersion = "475"`, `jackson-module-kotlin` 2.18.12 e
  `jackson-annotations` (`compileOnly`) 2.18.3.

## 5. O que a compilação não garante

O CI **nunca roda o plugin num Trino real**: `compat-jvm-plugins` executa só
`gradle -p trino-plugin test` (SPI como dependência de teste), e `compat-trino`
usa o conector Iceberg padrão do Trino, não o plugin. As verificações ao vivo
registradas na documentação foram manuais. Riscos a validar num servidor 475:

- **Serialização dos handles entre coordinator e worker.** Já houve dois bugs
  reais de Jackson + Kotlin nesse caminho (`JVM_INTEGRATION.md` §5A), e o Jackson
  do servidor muda de 2.15 para 2.18.
- **Semântica do `getTableHandle`:** devolver `null` para tabela inexistente, em
  vez de lançar exceção.
- **Classloader:** o `slf4j-api` já precisa ir no JAR. Conferir se algo mais
  mudou entre as versões.
- **Procedures (`CALL ailake.system.*`)** e **session properties**: APIs não
  apareceram nos erros de compilação, mas o comportamento precisa ser testado.

## 6. Plano

1. Branch com as 8 correções da §4 e `build.gradle.kts` atualizado; build com
   `jvmToolchain(17)`.
2. Rodar a demo existente (`tests/docker/demo/Dockerfile.trino`, hoje
   `FROM trinodb/trino:430`) com `trinodb/trino:475` e com a mais recente (480).
   Cobrir `SELECT` em `search`/`search_full`/`search_multimodal`, `INSERT`,
   `DELETE`, `ALTER TABLE ADD/RENAME COLUMN` e as procedures.
3. **Novo job de CI** que suba `trinodb/trino:475` com o plugin e a lib nativa e
   rode um smoke test SQL. É o que falta para não depender de verificação manual,
   e o que teria pego a quebra da 460.
4. Atualizar a documentação que fixa a 430:
   - `docs/guides/JVM_INTEGRATION.md` (tabela de versões, §5A, troubleshooting)
   - `docs/specs/JVM_PLUGINS.md`, `docs/specs/INTEGRATIONS.md`, `docs/WHY_AILAKE.md`
   - `docs/guides/DEMO_NOTEBOOKS.md`, `tests/docker/demo/Dockerfile.trino`,
     `tests/docker/compose-demo.yml`
5. CHANGELOG com a quebra de compatibilidade (mínimo Trino 475) e release do
   plugin (`release-trino.yml`).
6. Depois do merge em `main`, rodar `dependency-submission.yml`: os 15 alertas
   dispensados devem passar a `fixed`.

O `compat-trino` (conector Iceberg padrão, `trinodb/trino:436`) é independente do
plugin e pode continuar como está. Subir a versão dele exigiria trocar o catálogo
`hadoop`, que o Trino 460+ não aceita mais (comentário no `compat-heavy.yml`).

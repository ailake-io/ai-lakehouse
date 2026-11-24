# Versões e artefatos dos plugins

Cada plugin mantém sua própria versão. O Python acompanha a versão do core.
Os exemplos de instalação e os manifests são verificados por
`python3 scripts/plugins.py check`; referências de versão da documentação estão
registradas em `scripts/plugins.json` junto com os manifests.

| Componente | Tag de instalação | Artefatos esperados |
|---|---|---|
| Spark | [spark-v0.1.17](https://github.com/ailake-io/ai-lakehouse/releases/tag/spark-v0.1.17) | `ailake_jni-windows-x86_64.dll`, `libailake_jni-linux-x86_64.so`, `libailake_jni-macos-arm64.dylib`, `spark-plugin-0.1.17-plugin.jar` |
| Trino | [trino-v0.1.17](https://github.com/ailake-io/ai-lakehouse/releases/tag/trino-v0.1.17) | `ailake_jni-windows-x86_64.dll`, `libailake_jni-linux-x86_64.so`, `libailake_jni-macos-arm64.dylib`, `trino-plugin-0.1.17-plugin.jar` |
| Flink | [flink-v0.1.18](https://github.com/ailake-io/ai-lakehouse/releases/tag/flink-v0.1.18) | `ailake-flink-0.1.18-plugin.jar`, `ailake_jni-windows-x86_64.dll`, `libailake_jni-linux-x86_64.so`, `libailake_jni-macos-arm64.dylib` |
| Airflow provider | [airflow-v0.1.16](https://github.com/ailake-io/ai-lakehouse/releases/tag/airflow-v0.1.16) | `apache_airflow_providers_ailake-0.1.16-py3-none-any.whl`, `apache_airflow_providers_ailake-0.1.16.tar.gz` |
| Airbyte destination | [airbyte-v0.1.16](https://github.com/ailake-io/ai-lakehouse/releases/tag/airbyte-v0.1.16) | `airbyte_destination_ailake-0.1.16-py3-none-any.whl`, `airbyte_destination_ailake-0.1.16.tar.gz` |
| DuckDB extension | [duckdb-v0.1.17](https://github.com/ailake-io/ai-lakehouse/releases/tag/duckdb-v0.1.17) | `ailake-duckdb-v1.5.6.duckdb_extension`, `ailake-duckdb-v2.0.0.duckdb_extension` |
| C++ SDK | [cpp-v0.1.16](https://github.com/ailake-io/ai-lakehouse/releases/tag/cpp-v0.1.16) | `ailake-cpp-0.1.16-source.tar.gz` |
| Go SDK | [ailake-go/v0.1.18](https://github.com/ailake-io/ai-lakehouse/releases/tag/ailake-go/v0.1.18) | `ailake-go-0.1.18-source.tar.gz` |
| Python SDK | Acompanha a versão do core | PyPI: `ailake`; wheels Linux/macOS/Windows e sdist |
| Kof integration | [kof-v0.1.1](https://github.com/ailake-io/ai-lakehouse/releases/tag/kof-v0.1.1) | `ai-lake-kof-0.1.1-source.tar.gz` |

O manifesto descreve a versão preparada no código. Ao preparar uma próxima
versão, atualize os exemplos junto com o manifesto no mesmo PR. Os links da
nova tag ficam disponíveis quando a publicação termina. Para instalar uma
versão já publicada, confira a existência do release e de todos os artefatos.

Para auditar o que foi efetivamente publicado no GitHub e PyPI:

```bash
git fetch origin --tags
python3 scripts/release_audit.py
```

A auditoria verifica a tag, as versões nos arquivos fonte dessa tag, os nomes dos
artefatos registrados e, para os pacotes Python, a versão e distribuições no
PyPI. Ela falha se uma versão preparada ainda não foi publicada por completo.
O acesso à imagem Airbyte no GHCR depende da visibilidade e das permissões do
pacote; wheels e sdist podem ser conferidos pelo GitHub/PyPI separadamente.

Releases de plugins usam a versão preparada no PR. O workflow recusa publicar
outro número sem atualizar os manifests e a documentação. Reruns de versões
existentes preservam o commit original da tag, inclusive em releases históricos.
Veja [o guia de manutenção](../contributing/PLUGIN_MAINTENANCE.md).

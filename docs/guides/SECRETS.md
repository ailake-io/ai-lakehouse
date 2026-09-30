# Secrets e rotação de credenciais

O AI-Lake mantém credenciais em `secrecy::SecretString` no núcleo Rust. O
conteúdo não é incluído em `Debug` nem em mensagens de erro dos providers; os
buffers temporários usados para leitura de arquivos são zerados com `zeroize`.

## Referências suportadas

O CLI aceita referências rotativas para `--rest-token-ref` e
`--rest-oauth-client-secret-ref`:

```text
env://MINHA_VARIAVEL
file:///run/secrets/ailake-token
k8s:///var/run/secrets/ailake/token
k8s:///var/run/secrets/ailake#token        # arquivo JSON ou Secret montado
vault://kv/data/ailake#token
aws-sm://prod/ailake#token
```

Para Vault, informe `--vault-address` e `--vault-token-file` (modo `0600`), ou
use `AILAKE_VAULT_ADDR`/`AILAKE_VAULT_TOKEN`. Para AWS, o SDK usa a cadeia
normal de credenciais e `--secrets-aws-region`/`AWS_REGION` é opcional.

Exemplo com bearer token no Kubernetes:

```bash
ailake \
  --catalog rest \
  --rest-uri https://catalog.example/v1 \
  --rest-auth bearer \
  --rest-token-ref k8s:///var/run/secrets/ailake/token \
  --secrets-refresh-secs 60 \
  list default
```

O valor é carregado sob demanda e renovado após `--secrets-refresh-secs`. Em
OAuth2, a nova credencial é usada na próxima renovação do access token; o token
de acesso continua protegido e em cache até perto da expiração.

Para compilar apenas a biblioteca sem os SDKs opcionais, use os recursos
`vault` e `kubernetes`. O binário `ailake-cli` habilita também
`aws-secrets-manager`.

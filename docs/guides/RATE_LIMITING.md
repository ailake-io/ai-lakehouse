# Rate limiting distribuído e circuit breakers

O `ailake serve` oferece controle de tráfego compartilhado entre instâncias por
Redis/Valkey. Quando `--rate-limit-url` não é informado, o servidor reutiliza
`--cache-url`, se configurado; sem Redis/Valkey, o limite é local ao processo.

## Dimensões e quotas

Há duas classes de operação:

- **busca**: endpoints de busca e operações de indexação;
- **escrita**: ingestão, compactação e alteração de jobs.

Cada classe pode ser limitada por token bearer e por IP. Os contadores usam uma
janela fixa (60 segundos por padrão) e chaves derivadas por hash; o bearer token
não é gravado como texto no Redis.

Exemplo:

```bash
ailake serve default.table \
  --cache-url redis://redis:6379/0 \
  --rate-limit-url redis://redis:6379/1 \
  --rate-window-secs 60 \
  --search-quota-token 600 \
  --write-quota-token 60 \
  --search-quota-ip 1200 \
  --write-quota-ip 120
```

Quando a quota é excedida, a API retorna `429 Too Many Requests` e o header
`Retry-After` com o número de segundos até a próxima janela.

## Integração com gateway/API proxy

Por segurança, o servidor só considera `X-Forwarded-For` e `X-Real-IP` quando
`--trust-proxy-headers` está habilitado. Nesse modo, o primeiro endereço de
`X-Forwarded-For` é usado como IP do cliente:

```bash
ailake serve default.table \
  --auth-token-env AILAKE_AUTH_TOKEN \
  --rate-limit-url redis://redis:6379/1 \
  --trust-proxy-headers
```

Esse parâmetro deve ser usado apenas quando o processo está atrás de um proxy
que remove headers recebidos do cliente e os reescreve com a origem real. Sem
essa opção, a quota por IP não é aplicada; a quota por token continua ativa.

Se o Redis ficar indisponível, o padrão é **fail-open** para preservar a
disponibilidade. Para rejeitar requisições até a recuperação do Redis, use:

```bash
--rate-limit-fail-closed
```

## Circuit breakers

O catálogo e o object storage são protegidos por circuit breakers. Depois de
`--circuit-failure-threshold` falhas consecutivas (5 por padrão), novas
operações falham rapidamente com `503` durante
`--circuit-cooldown-secs` segundos (15 por padrão). Em seguida, uma requisição
de sondagem pode fechar o circuito quando a dependência se recuperar.

```bash
--circuit-failure-threshold 5 \
--circuit-cooldown-secs 15
```

Contenção de lock não é tratada como falha do object storage, evitando abrir o
circuito por concorrência normal de escritores.

## Métricas

O endpoint `/metrics` expõe:

- `ailake_http_rate_limited_total` — respostas HTTP limitadas;
- `ailake_rate_allowed_total` e `ailake_rate_limited_total` — decisões do
  limitador;
- `ailake_rate_backend_errors_total` — falhas do Redis/Valkey;
- `ailake_rate_redis_requests_total` — chamadas ao backend distribuído;
- `ailake_catalog_circuit_open` e `ailake_storage_circuit_open` — estado dos
  circuit breakers.

Em produção, recomenda-se usar Redis/Valkey compartilhado, definir limites por
token coerentes com o plano do cliente e alertar quando qualquer circuito
permanecer aberto ou quando `ailake_rate_backend_errors_total` crescer.

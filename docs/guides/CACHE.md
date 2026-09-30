# Cache de produção

O `ailake serve` usa um cache em duas camadas:

- memória local, compartilhada por todas as requisições do processo;
- Redis ou Valkey opcional, compartilhado entre instâncias.

As três partições são `query`, `metadata` e `index`. O limite configurado é
global para a memória local, somando todas as partições. Quando o limite é
atingido, as entradas mais antigas são removidas.

## Configuração

```bash
ailake serve namespace.tabela \
  --cache-url redis://127.0.0.1:6379/0 \
  --cache-max-bytes 268435456 \
  --cache-ttl-secs 2
```

As mesmas opções podem ser definidas com `AILAKE_CACHE_URL`,
`AILAKE_CACHE_MAX_BYTES` e `AILAKE_CACHE_TTL_SECS`. `redis://` e `rediss://`
(TLS) são aceitos pelo cliente Redis/Valkey.

## Invalidação e consistência

Resultados de busca e leituras de dados/índices carregam o `snapshot_id` na
chave. Ao observar um snapshot novo, as entradas locais de versões anteriores
são descartadas; chaves antigas no Redis expiram pelo TTL e nunca são usadas
para o snapshot novo. Escritas e compactações também removem a entrada de
metadados corrente no Redis.

Locks, leases, registros de jobs, cancelamentos e demais objetos de
coordenação nunca passam pelo cache.

O Redis é uma camada de aceleração: se estiver indisponível, a requisição
continua usando o storage primário e a falha aparece em
`ailake_cache_redis_errors_total`.

## Métricas

`GET /metrics` expõe hits, misses, inserts, evictions, invalidações, hits e
falhas do Redis, número de entradas e bytes locais em uso. As métricas
principais são:

- `ailake_cache_hits_total`
- `ailake_cache_misses_total`
- `ailake_cache_evictions_total`
- `ailake_cache_bytes_in_use`
- `ailake_cache_bytes_limit`
- `ailake_cache_redis_hits_total`
- `ailake_cache_redis_errors_total`

O cache de metadados corrente é intencionalmente limitado pelo TTL. Em uma
implantação com escritores externos ao `ailake serve`, use um TTL curto e
considere o snapshot da catalogação como a fonte de verdade; o cache de
resultados de busca permanece protegido pelo `snapshot_id`.

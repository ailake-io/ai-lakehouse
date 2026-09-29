# Coordenação multi-writer

Os backends S3, GCS e Azure usam objetos de lease sob o prefixo configurado do
store. A aquisição usa criação condicional; a recuperação de um lease expirado
usa atualização condicional baseada em `ETag`/versão. Isso impede que duas
instâncias considerem o mesmo lock adquirido.

Cada lease carrega:

- `owner`: identidade efêmera do processo;
- `fence`: token monotônico por lock;
- `expires_ms`: prazo de expiração.

Workers de indexação validam o fencing token antes de publicar progresso ou o
estado final. Se perderem o lease, a publicação é rejeitada e o worker deve
parar/repetir. O registry de jobs do `ailake serve` faz read-modify-write sob
`metadata/ailake_jobs.registry.lock`, portanto instâncias diferentes não
perdem atualizações concorrentes.

## Uma instância de `ailake serve`

O servidor adquire:

```text
metadata/ailake-serve/<namespace>/<table>.lock
```

Por isso duas instâncias podem servir tabelas diferentes no mesmo storage, mas
duas instâncias da mesma tabela não iniciam simultaneamente. O lease é renovado
a cada 30 segundos; se a renovação falhar, o servidor encerra graciosamente
para não continuar atendendo com uma posse perdida.

O backend local mantém o lock exclusivo por arquivo. Nesse modo o fencing
retorna o token compatível `0`; a coordenação entre processos continua sendo
garantida pelo `create_new` do arquivo.

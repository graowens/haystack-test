# Install the mistral library

```docker exec -it gra_ollama bash```

and

```ollama pull mistral```


Curl Demo

```aiignore
curl http://10.66.125.96:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "What is the capital of France?",
  "stream": false
}'
```

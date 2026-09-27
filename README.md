# Trabalho de Aprendizado por Reforço — horta comunitária

## Situação atual

Tema, DQN/PPO/A2C e possível grupo em dupla aguardam validação do professor. O enunciado exige aprovação do problema antes da implementação. Este material prepara o notebook e o protocolo; ainda não existe ambiente implementado nem resultado de treinamento.

## Arquivos

- `trabalho_horta.ipynb`: estrutura do relatório com textos e código de preparação integrados.
- `configs/experimentos.json`: 12 configurações candidatas, sementes e orçamentos propostos.
- `apresentacao/roteiro.md`: estrutura do vídeo de até três minutos.
- ZIPs originais: materiais das aulas, preservados na pasta.

## Próximos passos

1. Registrar resposta do professor e confirmar integrantes e prazo.
2. Fixar regras do MDP, coeficientes de recompensa e critérios de sucesso.
3. Implementar e validar ambiente/renderização.
4. Executar piloto; ajustar o orçamento conforme custo medido antes da busca.
5. Executar busca, registrar todas as configurações e selecionar pela validação.
6. Retreinar vencedoras, executar teste reservado e escrever resultados/conclusões.
7. Gravar apresentação com todos os membros, publicar no YouTube e incluir link.
8. Executar notebook completo, converter e revisar relatório.

## Abrir o notebook

Abra o Jupyter a partir desta pasta. As células atuais usam somente a biblioteca padrão do Python e não treinam agentes. A implementação futura exigirá Gymnasium, Stable-Baselines3, PyTorch e ferramentas de gráficos/renderização. Esses pacotes não foram instalados neste preparo. Escolher uma versão estável compatível e registrar as versões reais antes do treino.

Para o relatório final, com Jupyter/nbconvert instalado:

```powershell
jupyter nbconvert --to html trabalho_horta.ipynb
```

A versão exportada só será a entrega final depois de conter os experimentos reais, as conclusões e o link do vídeo.


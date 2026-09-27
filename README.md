# Irrigação inteligente de uma horta comunitária

Trabalho de Aprendizado por Reforço: ambiente próprio no Gymnasium e comparação de DQN, PPO e A2C com Stable-Baselines3.

## Aprovação e integrantes

Em 26/09/2026, o usuário instruiu considerar a proposta e os algoritmos aprovados e autorizou implementação e publicação dos avanços. Os nomes completos e o prazo de entrega precisam ser preenchidos no relatório. Não foi anexado um registro independente da resposta do professor.

## Ambiente

Quatro canteiros com necessidades diferentes disputam um reservatório de oito unidades. O agente observa umidade, saúde, água, clima e tempo restante; escolhe entre irrigar um dos quatro canteiros, reabastecer ou esperar. O clima segue uma cadeia de Markov. O episódio tem 60 turnos e pode terminar antes se todas as plantas morrerem. As regras são sintéticas e não representam manejo agronômico validado.

![Ambiente](reports/figures/ambiente.png)

## Estrutura

- `horta/env.py`: MDP, Gymnasium, renderização e heurística de referência.
- `horta/experiments.py`: treinos, avaliação, seleção e registro retomável.
- `horta/analysis.py`: agregação, gráficos e execução ilustrativa.
- `configs/experimentos.json`: 12 configurações, orçamentos e sementes.
- `tests/test_environment.py`: verificações da API, água, terminalidade e sementes.
- `results/`: dados reais; o piloto de diagnóstico fica separado.
- `trabalho_horta.ipynb`: notebook integrado ao relatório.
- `scripts/build_report.py`: atualiza/executa o notebook e exporta HTML.
- `apresentacao/roteiro.md`: roteiro da apresentação.
- `requirements-lock.txt`: versões usadas (Python 3.12, Windows, PyTorch CPU).

ZIPs das aulas, ambiente virtual, modelos e vídeos brutos ficam fora dos novos commits. Os ZIPs enviados inicialmente ainda constam no histórico antigo.

## Instalação

Na raiz do projeto, com Python 3.12:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

O lock registra o ambiente utilizado. Para outro sistema operacional, instalar as dependências diretas de `requirements.txt` e registrar um novo lock.

## Verificações e experimentos

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m horta.experiments --stage pilot
.venv/Scripts/python.exe -m horta.experiments --stage tuning
.venv/Scripts/python.exe -m horta.experiments --stage final
.venv/Scripts/python.exe -m horta.experiments --stage baselines
.venv/Scripts/python.exe -m horta.analysis
```

É possível executar a busca por algoritmo em processos separados usando `--algorithm DQN`, `--algorithm PPO` e `--algorithm A2C`. Cada processo usa uma thread de PyTorch. Após concluir os três, executar a etapa final sem filtro para selecionar configurações com todos os dados.

O runner retoma execuções concluídas sem repetir o treino e recusa mudanças no protocolo dessas execuções. Os modelos locais ficam em `models/`; dados e parâmetros vão para o Git. Para recriar modelos em outro clone, arquivar os resultados existentes antes de refazer as execuções.

## Relatório e apresentação

```powershell
.venv/Scripts/python.exe scripts/build_report.py
```

O relatório carrega dados reais já produzidos e contém comandos explícitos para treinamento, evitando repetir horas de treino ao abrir o notebook. O HTML gerado fica em `reports/relatorio_horta.html`. O vídeo ilustrativo fica em `videos/execucao_ppo.mp4`; ele não substitui a apresentação com fala de todos os integrantes, de até três minutos, publicada no YouTube. Incluir o link e preencher os nomes antes da entrega.

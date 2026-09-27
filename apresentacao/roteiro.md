# Apresentação — roteiro com resultados reais

Integrantes: Nicolas Cussioli Raimundo, Marcelo Fontana.

Meta: 2min50s, com 10 segundos de margem. Ensaiar com cronômetro. Todos os integrantes devem falar. A divisão abaixo considera Nicolas e Marcelo; se houver outros integrantes, redistribuir os trechos.

## 0:00–1:10 — Nicolas: problema e MDP

Olá! Somos Nicolas e Marcelo. Criamos uma horta virtual para estudar como manter quatro canteiros saudáveis usando pouca água.

No ambiente Gymnasium, o agente observa umidade, saúde das plantas, água disponível, clima e tempo restante. Ele pode irrigar um canteiro, reabastecer ou esperar.

O clima muda de forma probabilística. A chuva aumenta a umidade; evaporação e drenagem reduzem a água no solo. Cada canteiro tem uma faixa ideal diferente. Seca e excesso de água prejudicam as plantas.

A recompensa favorece a saúde e penaliza consumo de água, reabastecimentos e mortes. O episódio dura até sessenta turnos. O tempo faz parte do estado porque influencia as decisões. As regras são didáticas, sem validação agronômica.

**Visual:** renderização da horta e uma lista curta de estado, ações e recompensa.

## 1:10–2:30 — Marcelo: experimentos e resultados

Comparamos DQN, PPO e A2C, com quatro configurações por algoritmo e três sementes de treinamento. Escolhemos a melhor pela média da validação.

Depois, retreinamos com cinco sementes novas. Cada modelo foi avaliado em cem episódios separados, sem continuar aprendendo. Comparamos também com ações aleatórias e uma regra que irriga o canteiro com maior necessidade.

PPO teve o maior retorno médio entre os algoritmos: 55.9, com desvio de 0.3. Seu sucesso médio foi de 94.2 por cento. Sucesso exige terminar com todas as plantas vivas e saúde média de pelo menos cinquenta por cento.

A heurística teve retorno 55.0, e a política aleatória, 13.6. O gráfico compara os métodos. Olhamos também saúde e água: gastar pouco pode significar deixar plantas morrerem.

**Visual:** gráfico `reports/figures/comparacao.png`, mais uma execução curta de `apresentacao/execucao_ppo.mp4`. Valores completos para consultar: PPO: 55.90 ± 0.26; A2C: 55.14 ± 0.61; DQN: 54.42 ± 0.84.

## 2:30–2:50 — conclusão dividida entre integrantes

**Nicolas:** Conseguimos implementar o MDP e comparar três algoritmos com parâmetros e avaliações registrados.

**Marcelo:** Os resultados valem para este simulador. Como próximos passos, podemos usar sensores com ruído, outras plantas e maior busca de hiperparâmetros.

## Publicação

- Gravar fala real de todos os integrantes e conferir duração máxima de três minutos.
- Publicar no YouTube; o MP4 ilustrativo não substitui a apresentação.
- Incluir o link em `configs/entrega.json` e gerar o relatório novamente.
- Divulgação em rede social é opcional; inserir o link se realizada.

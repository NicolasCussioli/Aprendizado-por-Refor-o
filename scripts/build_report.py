"""Gera notebook literate, executa as células de análise e exporta HTML."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horta.experiments import collect_runs, configuration


def build():
    config = configuration()
    cells = []
    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbformat.v4.new_code_cell(text))

    final_path = ROOT / "results/final_summary.csv"
    finished = final_path.exists()
    delivery = json.loads((ROOT / "configs/entrega.json").read_text(encoding="utf-8"))
    youtube = f"[Apresentação no YouTube]({delivery['youtube']})" if delivery["youtube"] else "[incluir link após gravação com a fala de todos os integrantes]"
    social = f"[Divulgação]({delivery['divulgacao']})" if delivery["divulgacao"] else "[opcional]"
    md(f"""# Irrigação inteligente de uma horta comunitária

**Trabalho de Aprendizado por Reforço — notebook e relatório**

Integrantes: {', '.join(delivery['integrantes'])}. Professor: {delivery['professor']}. Prazo: {delivery['prazo']}.

Em 26/09/2026, o usuário instruiu considerar tema, algoritmos e grupo aprovados. Essa informação autoriza a continuidade do projeto; a resposta original do professor não foi anexada.

**Apresentação no YouTube:** {youtube}. Divulgação em rede social: {social}.

Este relatório contém código executável e dados reais. O treinamento é executado pelo módulo de experimentos, com comando explícito, e as células seguintes carregam seus registros sem repetir os treinamentos.""")
    md("""## 1. Introdução

Uma horta comunitária precisa distribuir água entre canteiros com necessidades diferentes. Irrigar cedo pode desperdiçar água caso chova; esperar demais pode comprometer a saúde das plantas. Reabastecer ocupa um turno e também permite que o solo continue secando. Esse cenário constitui um problema de decisão sequencial com consequências futuras e incerteza.

Construímos um simulador didático próprio no Gymnasium e comparamos DQN, PPO e A2C, da Stable-Baselines3. O ambiente combina quatro canteiros, reservatório limitado, saúde persistente, falta e excesso de água e clima probabilístico. A comparação utiliza uma busca controlada de hiperparâmetros, várias sementes de treinamento e episódios separados de validação e teste.

O cenário foi elaborado para o trabalho. Não afirmamos ineditismo na literatura nem fidelidade agronômica: todos os coeficientes e unidades são sintéticos.""")
    if finished:
        import pandas as pd
        summary = pd.read_csv(final_path)
        rl = summary[summary.training_seeds > 0]
        best = rl.sort_values("episode_return_mean", ascending=False).iloc[0]
        heuristic = summary[summary.algorithm == "heuristic"].iloc[0]
        md(f"Nas condições avaliadas, {best.algorithm} obteve o maior retorno médio entre os métodos de RL: **{best.episode_return_mean:.2f} ± {best.episode_return_std:.2f}**, com sucesso em **{best.success_mean:.1%}** dos episódios, em média entre treinamentos. A heurística apresentou retorno **{heuristic.episode_return_mean:.2f}**. Esses resultados se referem a este simulador e protocolo; não demonstram superioridade geral de um algoritmo.")
    else:
        md("**Resultados em andamento:** o ambiente e o piloto já foram verificados. As descobertas finais serão acrescentadas após concluir a busca e o teste reservado.")

    md(r"""## 2. Modelagem do problema

### 2.1 Estados, ações e horizonte

O MDP é $M=(S,A,P,R,H)$, com horizonte $H=60$. O estado contém quatro umidades $u_i$, quatro saúdes $h_i$, água $w$, regime climático $c$ e tempo restante $\tau$. Umidade e saúde pertencem a $[0,1]$, e $w\in[0,8]$. As ações são: irrigar canteiro 1, 2, 3 ou 4; reabastecer; esperar.

A observação é um vetor `Box(0,1,(13,))`: quatro umidades, quatro saúdes, água dividida por oito, três indicadores one-hot do clima e tempo restante dividido por 60. As ações usam `Discrete(6)`. A saúde registra os efeitos de estresse anteriores; o tempo restante permite representar o horizonte finito. Ambos são necessários para o estado conter as informações que influenciam as próximas transições e recompensas.

As condições iniciais são $u_i\sim U(0{,}4;0{,}7)$, $h_i=1$, $w\sim U(4;8)$ e clima seco/ameno/chuvoso com probabilidades $(0{,}50;0{,}35;0{,}15)$. O episódio termina ao completar 60 turnos ou morrerem todas as plantas. Como o horizonte pertence ao MDP e aparece no estado, seu fim gera `terminated=True`, e não `truncated=True`.

### 2.2 Clima e transições

A matriz abaixo usa a ordem seco, ameno, chuvoso tanto nas linhas como nas colunas:

$$P(c_{t+1}\mid c_t)=\begin{bmatrix}0{,}80&0{,}18&0{,}02\\0{,}25&0{,}55&0{,}20\\0{,}10&0{,}30&0{,}60\end{bmatrix}.$$

| Parâmetro | Seco | Ameno | Chuvoso |
|---|---:|---:|---:|
| Evaporação base | 0,018 | 0,012 | 0,006 |
| Probabilidade de chuva | 0,05 | 0,15 | 0,65 |
| Quantidade quando chove | 0,02 | 0,04 | 0,08 |

| Canteiro | Umidade mínima ideal | Máxima ideal | Multiplicador de evaporação | Drenagem por turno |
|---|---:|---:|---:|---:|
| 1 | 0,30 | 0,70 | 0,8 | 0,010 |
| 2 | 0,38 | 0,75 | 1,0 | 0,012 |
| 3 | 0,34 | 0,68 | 1,2 | 0,016 |
| 4 | 0,42 | 0,80 | 1,1 | 0,009 |

Uma irrigação usa $q=\min(1,w)$ unidades e acrescenta $0{,}20q$ à umidade do canteiro vivo selecionado. Sem água ou com uma planta morta, a irrigação não tem efeito e recebe penalidade. Reabastecer coloca oito unidades no reservatório e consome o turno; a fonte externa de reposição é ilimitada, mas requer essa ação. Água em excesso no solo é perdida pelo recorte em umidade máxima um, continuando a contar como consumida.

Para cada canteiro, a evaporação é a base climática multiplicada pela demanda e por um fator independente $U(0{,}9;1{,}1)$. A nova umidade é a anterior mais irrigação e chuva menos evaporação e drenagem, limitada a $[0,1]$. Primeiro ocorrem ação e efeitos do clima atual; em seguida são atualizadas as saúdes, sorteado o clima seguinte e decrementado o tempo restante.

### 2.3 Saúde e recompensa

Sejam $d_i^- = \max(L_i-u_i',0)$ e $d_i^+ = \max(u_i'-U_i,0)$. Uma planta viva ganha 0,015 de saúde quando a umidade está na faixa ideal, perde $0{,}04+0{,}30d_i^-$ quando seca e perde $0{,}02+0{,}15d_i^+$ quando encharcada. A saúde é limitada a $[0,1]$; chegar a zero é irreversível.

$$r_t=\overline{h}_{t+1}-0{,}10q_t-2d_t-0{,}05\mathbb{1}[a_t=\text{reabastecer}]-0{,}10\mathbb{1}[\text{irrigação inválida}],$$

em que $d_t$ conta apenas as plantas que morreram nesse turno. O termo de saúde incentiva manter todos os canteiros vivos; os custos de água e reabastecimento evitam irrigar indiscriminadamente. A saúde média inclui plantas mortas. Esses pesos foram fixados antes da busca de hiperparâmetros e não ajustados usando os resultados de teste.

O gerador `self.np_random` controla a aleatoriedade. A quantidade e a ordem dos sorteios do clima não dependem da ação, permitindo comparar políticas sob os mesmos eventos externos quando usam a mesma semente. As regras do simulador e o regime atual são suficientes para caracterizar a distribuição do próximo estado; o histórico não acrescenta informação necessária.

### 2.4 Implementação

A célula seguinte contém o código real de `horta/env.py`, incluído automaticamente para manter texto e implementação juntos. A renderização oferece modos `rgb_array`, `human` e `ansi`; a imagem é usada para compreensão e apresentação, enquanto a política recebe o vetor numérico.""")
    code((ROOT / "horta/env.py").read_text(encoding="utf-8"))
    code("""from IPython.display import display, Image, Code
from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3.common.env_checker import check_env as sb3_check

gym_check(HortaEnv(), skip_render_check=True)
sb3_check(HortaEnv(), warn=True)
print('API do Gymnasium e compatibilidade com SB3 verificadas.')
display(Image(filename='reports/figures/ambiente.png'))""")
    md("Além da API, seis testes verificam conservação de água, reabastecimento, morte irreversível, encerramento no horizonte, limites das observações, formato da imagem e reprodução do clima por semente independentemente das ações. Executar `python -m unittest discover -s tests -v` para reproduzi-los.")

    md("""## 3. Metodologia de experimentos

### 3.1 Algoritmos

| Método | Princípio | Justificativa |
|---|---|---|
| DQN | Rede para valores de ação, replay buffer e rede alvo; off-policy | Amplia o Q-learning tabular trabalhado nas aulas |
| PPO | Ator e crítico com objetivo de política sujeito a clipping; on-policy | Método usado no módulo 3; referência para aprendizagem de política |
| A2C | Ator e crítico com atualizações por vantagem; on-policy | Compara outro método de política e estimação de valor |

Os três suportam ações discretas e usam `MlpPolicy`. O DQN usa camadas ocultas `[64,64]`; PPO e A2C usam `[64,64]` no ator e no crítico. Não se trata de uma igualdade de número de parâmetros: as famílias têm estruturas distintas. Todos usam CPU, um ambiente por treino e uma thread de PyTorch por processo. O tempo de treino será reportado, mas a execução simultânea pode introduzir variação por disputa dos recursos do computador.

### 3.2 Busca de hiperparâmetros

Propusemos uma base e três variações por algoritmo; cada variação altera um parâmetro em relação à base. A busca não é exaustiva nem um fatorial completo. Os parâmetros de interesse são a taxa de aprendizado, o desconto e um parâmetro específico: duração da exploração no DQN, clipping no PPO e entropia no A2C.

| Algoritmo | Base | Variação 1 | Variação 2 | Variação 3 |
|---|---|---|---|---|
| DQN | lr=0,0001; gamma=0,99; exploração=0,30 | lr=0,0003 | gamma=0,95 | exploração=0,50 |
| PPO | lr=0,0003; gamma=0,99; clip=0,20 | lr=0,0001 | gamma=0,95 | clip=0,10 |
| A2C | lr=0,0007; gamma=0,99; entropia=0,01 | lr=0,0003 | gamma=0,95 | entropia=0,05 |

`learning_rate` controla o tamanho das atualizações; `gamma`, o peso de retornos futuros. No DQN, os parâmetros de buffer e de exploração regulam armazenamento, reutilização e ações aleatórias. No PPO, `n_steps`, `batch_size` e `n_epochs` regulam coleta e otimização; `gae_lambda` controla a estimativa de vantagem. Em PPO/A2C, `ent_coef` estimula diversidade da política. No A2C, `vf_coef` pesa a perda do crítico e `use_rms_prop` seleciona o otimizador. A célula seguinte mostra todas as configurações propostas; os arquivos `run.json` registram parâmetros efetivos e versões utilizadas.""")
    code("""from pathlib import Path
import json, inspect
import pandas as pd
from horta.experiments import candidates, configuration, collect_runs, evaluate, run_trial

config = configuration()
rows = [{'algorithm': a, 'config_id': c, **p} for (a,c), p in candidates(config).items()]
display(pd.DataFrame(rows)[['algorithm','config_id','learning_rate','gamma']])
for algorithm, specification in config['algorithms'].items():
    print(algorithm, '— demais parâmetros da base:')
    print(json.dumps(specification['base'], indent=2))
print('Sementes:', json.dumps(config['seeds']))
print('Orçamentos:', config['budgets'])
print('Versões:', json.loads(next(Path('results/pilot').glob('*/run.json')).read_text(encoding='utf-8'))['dependencies'])""")
    md("""### 3.3 Protocolo, métricas e seleção

1. Piloto: configuração base de cada algoritmo, semente 7 e 20.480 interações. Serve para verificar o pipeline e medir custo; não decide a configuração vencedora. O orçamento originalmente proposto foi mantido.
2. Busca: 12 configurações × 3 sementes de treino (11, 22, 33), com 102.400 interações por execução. Avaliar o modelo do fim do orçamento em 30 episódios de validação (sementes 1000 a 1029).
3. Seleção por algoritmo: maior média dos retornos de validação, primeiro calculada por semente de treinamento e depois entre as três sementes. Em empate numérico, usar menor consumo médio de água. Não escolher o melhor checkpoint intermediário.
4. Final: treinar do zero a configuração selecionada de cada algoritmo com sementes 101 a 105 e 307.200 interações. Cada modelo é avaliado em 100 episódios de teste (sementes 2000 a 2099), com `deterministic=True`, sem aprendizado.
5. Referências: política aleatória, com gerador de ações separado do clima; e regra que irriga o maior déficit relativo ao limiar mínimo mais 0,07, reabastece quando necessário e espera quando nenhum canteiro precisa de água. Avaliar no mesmo teste.

O orçamento total do protocolo é 8.355.840 interações de treino, em 54 execuções, fora o diagnóstico inicial descartado. Os valores são múltiplos dos rollouts usados no PPO/A2C. Os registros permitem conferir passos efetivamente realizados. As sementes de validação não são usadas para medir o desempenho final; as sementes de teste não são usadas para escolher hiperparâmetros.

| Métrica | Definição |
|---|---|
| Retorno episódico | Soma não descontada das recompensas; métrica principal |
| Sobrevivência final | Fração de canteiros com saúde maior que zero |
| Saúde final média | Média da saúde dos quatro canteiros no fim |
| Água usada | Soma da água efetivamente aplicada; unidades sintéticas |
| Sucesso | Chegar ao turno 60 com todos vivos e saúde média ≥ 0,5 |
| Custo | Tempo e interações reais do treino |

O resultado de cada modelo é a média dos seus 100 episódios. O relatório apresenta a média e o desvio padrão amostral entre as cinco sementes de treinamento. Episódios de um único modelo não são tratados como treinamentos independentes. As referências fixas não possuem desvio entre treinamentos; uma ausência de barra para elas não indica ausência de incerteza entre cenários.

### 3.4 Código de treinamento e avaliação

O módulo executa cada tentativa com ambiente separado, salva o modelo localmente e os resultados por episódio em CSV, e registra falhas. A retomada recusa alterações de código/protocolo para não misturar dados incompatíveis. A célula abaixo mostra o código usado. A execução dos experimentos é opt-in: abrir o notebook para revisar o relatório não reinicia horas de treinamento.""")
    code("display(Code(inspect.getsource(evaluate), language='python'))\ndisplay(Code(inspect.getsource(run_trial), language='python'))")
    code("""import subprocess, sys
REEXECUTAR_TREINAMENTOS = False
if REEXECUTAR_TREINAMENTOS:
    subprocess.run([sys.executable, '-m', 'horta.experiments', '--stage', 'all'], check=True)
else:
    print('Treinamentos não reiniciados. Carregando os dados reais salvos em results/.')""")
    md("""### 3.5 Piloto, dificuldades e reprodutibilidade

No diagnóstico inicial, DQN e PPO concluíram; o registro do A2C falhou porque sua inicialização modificou `policy_kwargs`, incluindo uma classe de otimizador que não era serializável em JSON. O runner foi corrigido para entregar uma cópia profunda dos parâmetros ao algoritmo. Os arquivos dessa tentativa foram preservados em `results/pilot_diagnostico`, separados das comparações. O piloto dos três métodos foi refeito. A fonte da renderização também foi corrigida para suportar acentos.

O piloto verificado abaixo não serve como resultado final nem foi usado para escolher hiperparâmetros. Todas as configurações planejadas da busca serão apresentadas, inclusive as de baixo desempenho. Versões, sementes, código, orçamento, duração e status ficam em cada `run.json`; retornos de treino ficam em `monitor.csv` e avaliação por episódio em `evaluation.csv`. Um modelo final de cada algoritmo, sempre da semente 101 fixada antes da comparação, acompanha o repositório em `artifacts/models/`; os demais são artefatos locais regeneráveis. O lock de dependências registra o ambiente de execução.

Na execução paralela, as sementes finais de um algoritmo são agendadas assim que suas 12 tentativas de busca terminam. A seleção de cada algoritmo é independente da dos outros e permanece restrita à validação; candidatos e orçamentos foram fixados antes de observar o teste. São usados no máximo cinco processos finais simultâneos, além dos processos de busca ainda ativos. O protocolo estatístico não depende da ordem de execução.""")
    code("display(collect_runs('pilot').round(4))")

    md("## 4. Resultados\n\n### 4.1 Busca e escolha de configurações\n\nA tabela apresenta as tentativas efetivamente concluídas. Na versão final, cada configuração terá três treinamentos; uma busca incompleta não permite selecionar vencedores.")
    code("""tuning = collect_runs('tuning')
if tuning.empty:
    print('Busca ainda sem execuções concluídas.')
else:
    display(tuning.round(4))
    display(tuning.groupby(['algorithm','config_id']).agg(
        seeds=('training_seed','count'), retorno=('episode_return','mean'),
        desvio=('episode_return','std'), agua=('water_used','mean'),
        sucesso=('success','mean'), segundos=('train_seconds','mean')).round(4))""")
    if finished:
        winners = json.loads((ROOT / "results/winners.json").read_text(encoding="utf-8"))
        md("Seleção feita exclusivamente pela validação: " + "; ".join(f"**{a}: `{c}`**" for a,c in winners.items()) + ". As demais configurações permanecem nas tabelas para justificar a escolha. O vencedor na validação não precisa obter o mesmo desempenho após novo treinamento com sementes diferentes.")
        md("### 4.2 Teste final e referências\n\nResultados em episódios reservados, após retreinamento das configurações selecionadas. As colunas de desvio dos métodos de RL medem dispersão entre cinco treinamentos; nas referências são ausentes.")
        code("""summary = pd.read_csv('results/final_summary.csv')
def mean_sd(row, metric, precision=2, percent=False):
    factor = 100 if percent else 1
    mean, sd = row[metric+'_mean'] * factor, row[metric+'_std'] * factor
    suffix = '%' if percent else ''
    if pd.isna(sd):
        return f'{mean:.{precision}f}{suffix}'
    return f'{mean:.{precision}f} ± {sd:.{precision}f}{suffix}'
display(pd.DataFrame([{
    'Algoritmo': row.algorithm, 'Configuração': row.config_id,
    'Retorno': mean_sd(row,'episode_return'),
    'Sobrevivência': mean_sd(row,'survival_fraction',1,True),
    'Saúde': mean_sd(row,'final_mean_health',3),
    'Água': mean_sd(row,'water_used'),
    'Sucesso': mean_sd(row,'success',1,True),
} for _,row in summary.iterrows()]))
display(summary[['algorithm','training_seeds','train_seconds_mean','train_seconds_std']].round(2))
display(Image(filename='reports/figures/comparacao.png'))
display(Image(filename='reports/figures/agua_saude.png'))
display(Image(filename='reports/figures/aprendizado.png'))""")
        rl = summary[summary.training_seeds > 0].sort_values("episode_return_mean", ascending=False)
        best = rl.iloc[0]
        for _, row in rl.iterrows():
            md(f"**{row.algorithm} (`{row.config_id}`):** retorno {row.episode_return_mean:.2f} ± {row.episode_return_std:.2f}; sobrevivência {row.survival_fraction_mean:.1%}; saúde final {row.final_mean_health_mean:.3f}; água {row.water_used_mean:.2f}; sucesso {row.success_mean:.1%}. O desvio do retorno indica variação entre os treinamentos, além da diversidade dos episódios.")
        random = summary[summary.algorithm == "random"].iloc[0]
        heuristic = summary[summary.algorithm == "heuristic"].iloc[0]
        md(f"A referência aleatória obteve retorno {random.episode_return_mean:.2f}, enquanto a heurística obteve {heuristic.episode_return_mean:.2f}. O melhor método de RL ficou {best.episode_return_mean - heuristic.episode_return_mean:+.2f} pontos em relação à heurística e {best.episode_return_mean - random.episode_return_mean:+.2f} em relação à política aleatória. Essa comparação explicita se o aprendizado acrescenta benefício sobre uma regra baseada no conhecimento do simulador. Não foi aplicado teste de significância; a ordenação é descritiva.")
        md("As curvas exibem retornos coletados durante o treino, incluindo a exploração, e não os retornos determinísticos do teste. A suavização usa 100 episódios; as linhas são médias entre sementes em uma grade comum de interações e as faixas representam um desvio padrão. Dados brutos são mantidos nos arquivos Monitor. Consumir pouca água só é uma vantagem se a saúde também for preservada.")
        md("### 4.3 Execução ilustrativa\n\nPPO, primeira semente de treino (101) e primeira semente de teste (2000) foram definidos para a demonstração. A trajetória não foi escolhida por apresentar o melhor resultado e não substitui as médias. A animação acompanha o repositório, e o MP4 para edição fica localmente em `videos/execucao_ppo.mp4`.")
        code("display(Image(filename='reports/figures/execucao_ppo.gif'))\ndisplay(pd.read_csv('results/demo_trajectory.csv').head(10))")
        md(f"## 5. Conclusões\n\nO ambiente implementado permite relacionar o MDP a uma tarefa de gestão de recursos e comparar três métodos profundos. No protocolo executado, {best.algorithm} apresentou o maior retorno médio entre os algoritmos de RL. O sucesso médio foi de {best.success_mean:.1%}; esse indicador deve ser lido junto à dispersão do retorno, à saúde e ao consumo. A heurística atingiu retorno {heuristic.episode_return_mean:.2f}, oferecendo uma referência de solução por conhecimento das regras. Os resultados descrevem este cenário, sem estabelecer uma classificação geral de algoritmos.")
    else:
        md("**Busca e teste em andamento.** Não há conclusões finais nesta versão; o programa de agregação exige completar todas as sementes antes de gerar o resumo final.")
        md("## 5. Conclusões provisórias\n\nA implementação e o protocolo estão preparados; as conclusões de desempenho dependem dos experimentos completos.")
    md("""### Limitações, dificuldades e trabalhos futuros

O clima e as equações de saúde foram escolhidos para produzir um problema didático, sem calibração empírica. A política observa o estado completo e a reposição de água externa é ilimitada, embora exija uma ação. A busca tem somente quatro configurações por algoritmo e não explora conjuntamente todos os parâmetros; a arquitetura foi fixada. Cinco sementes finais ajudam a revelar variação, mas não eliminam incerteza. O ajuste dos pesos de recompensa e da arquitetura não foi incluído nos experimentos.

O registro de parâmetros do A2C e os acentos da renderização exigiram correção no piloto. A separação entre sementes de treino, validação e teste, a preservação de tentativas e a distinção entre episódios e treinamentos foram cuidados essenciais na organização do trabalho. A execução de treinos em paralelo também limita comparações estritas de tempo de CPU.

Como continuidade, propomos avaliar sensores ruidosos, reservatório captando chuva, escassez real da fonte de reposição, outras espécies, maior busca de hiperparâmetros e testes estatísticos com mais sementes. Esses itens são extensões, não funcionalidades já implementadas.

## 6. Apresentação e entrega

A apresentação deve durar no máximo três minutos, com todos os integrantes falando, e estar no YouTube. O roteiro em `apresentacao/roteiro.md` prioriza MDP e resultados. A animação gerada é material de apoio e não substitui a apresentação dos integrantes. Preencher nomes, professor, prazo e link antes da entrega. A divulgação em redes sociais é opcional, com link no relatório se realizada.

## Referências

- Materiais da disciplina: notebooks dos módulos 1 a 4. O módulo 3 exemplifica PPO; os módulos 1 e 4 tratam de ambientes próprios.
- [Gymnasium: criação de ambiente](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
- [Stable-Baselines3: repositório](https://github.com/DLR-RM/stable-baselines3) e [orientações de experimentação](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
- [DQN: documentação](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html); Mnih et al., [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), 2013.
- [PPO: documentação](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html); Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
- [A2C: documentação](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html). A família de métodos por vantagem com ator e crítico é apresentada por Mnih et al. em [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), 2016; o A2C usado aqui é a variante síncrona da SB3, não o A3C assíncrono do artigo.
""")
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nbformat.write(nb, ROOT / "trabalho_horta.ipynb")
    return nb


def main():
    os.chdir(ROOT)
    nb = build()
    kernel = ROOT / ".venv/share/jupyter/kernels/horta"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (horta)", "language": "python"}), encoding="utf-8")
    for variable, name in [("JUPYTER_RUNTIME_DIR", "runtime"), ("IPYTHONDIR", "ipython"), ("TEMP", "tmp"), ("TMP", "tmp")]:
        directory = ROOT / ".venv" / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    import tempfile
    tempfile.tempdir = str(ROOT / ".venv/tmp")
    os.environ["JUPYTER_PATH"] = str(ROOT / ".venv/share/jupyter")
    NotebookClient(nb, timeout=180, kernel_name="horta", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, ROOT / "trabalho_horta.ipynb")
    html, _ = HTMLExporter().from_notebook_node(nb)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/relatorio_horta.html").write_text(html, encoding="utf-8")
    count_tuning, count_final = len(collect_runs("tuning")), len(collect_runs("final"))
    delivery = json.loads((ROOT / "configs/entrega.json").read_text(encoding="utf-8"))
    checklist = f"""# Checklist do trabalho

- [x] Integrantes: {', '.join(delivery['integrantes'])}; dupla considerada aprovada conforme instrução do usuário.
- [x] Proposta e algoritmos considerados aprovados conforme instrução do usuário.
- [x] MDP documentado com estados, ações, transições, recompensa e horizonte.
- [x] Ambiente próprio implementado no Gymnasium.
- [x] Renderização rgb_array, human e ansi.
- [x] API do Gymnasium/SB3 e seis verificações do ambiente.
- [x] DQN, PPO e A2C integrados; piloto dos três concluído.
- [{'x' if count_tuning == 36 else ' '}] Busca de hiperparâmetros: {count_tuning}/36 treinamentos concluídos, 12 configurações e 3 sementes por configuração.
- [{'x' if count_final == 15 else ' '}] Retreinamento final: {count_final}/15 treinamentos concluídos, 5 sementes por algoritmo.
- [{'x' if (ROOT / 'results/heuristic.csv').exists() else ' '}] Referências aleatória e heurística avaliadas em teste.
- [{'x' if (ROOT / 'results/final_summary.csv').exists() else ' '}] Resultados, gráficos e comparação de todas as configurações.
- [x] Notebook com texto e código integrados, executado sem erros.
- [x] Notebook convertido para relatório HTML.
- [{'x' if (ROOT / 'results/final_summary.csv').exists() else ' '}] Discussão de resultados e conclusões preenchidas com dados reais.
- [{'x' if (ROOT / 'artifacts/models/manifest.json').exists() else ' '}] Modelos reproduzíveis e execução ilustrativa.
- [x] Roteiro de apresentação preparado.
- [ ] Gravar apresentação de até 3 minutos com fala de todos os integrantes.
- [{'x' if delivery['youtube'] else ' '}] Publicar no YouTube e incluir link no relatório.
- [{'x' if '[preencher]' not in delivery['professor'] else ' '}] Preencher nome do professor.
- [{'x' if '[preencher]' not in delivery['prazo'] else ' '}] Confirmar data de entrega e entregar no prazo.
- [{'x' if delivery['divulgacao'] else ' '}] Opcional: divulgar em rede social e incluir link.

Editar identificação e links em `configs/entrega.json` e executar `python scripts/build_report.py` para atualizar os documentos.
"""
    (ROOT / "CHECKLIST.md").write_text(checklist, encoding="utf-8")
    if (ROOT / "results/final_summary.csv").exists():
        import pandas as pd
        summary = pd.read_csv(ROOT / "results/final_summary.csv")
        rl = summary[summary.training_seeds > 0].sort_values("episode_return_mean", ascending=False)
        best = rl.iloc[0]
        heuristic = summary[summary.algorithm == "heuristic"].iloc[0]
        random = summary[summary.algorithm == "random"].iloc[0]
        ranking = "; ".join(f"{row.algorithm}: {row.episode_return_mean:.2f} ± {row.episode_return_std:.2f}" for _, row in rl.iterrows())
        script = f"""# Apresentação — roteiro com resultados reais

Integrantes: {', '.join(json.loads((ROOT / 'configs/entrega.json').read_text(encoding='utf-8'))['integrantes'])}.

Meta: 2min50s, com 10 segundos de margem. Ensaiar com cronômetro. Todos os integrantes devem falar. A divisão abaixo considera Nicolas e Marcelo; se houver outros integrantes, redistribuir os trechos.

## 0:00–1:10 — Nicolas: problema e MDP

Olá! Somos Nicolas e Marcelo. Nosso trabalho estuda a irrigação de uma horta comunitária usando aprendizado por reforço. O desafio é manter quatro canteiros saudáveis sem gastar água desnecessariamente.

Criamos um ambiente no Gymnasium. O agente observa a umidade e a saúde de cada canteiro, a água disponível, o clima e o tempo restante. Em cada turno, ele pode irrigar um canteiro, reabastecer o reservatório ou esperar.

O clima muda de forma probabilística. A chuva aumenta a umidade, enquanto evaporação e drenagem reduzem a água no solo. Cada canteiro tem uma faixa ideal diferente. Tanto a seca como o excesso de água prejudicam sua saúde.

A recompensa favorece plantas saudáveis e penaliza o uso de água, reabastecimentos e mortes. Cada episódio dura até sessenta turnos. Incluímos o tempo no estado porque ele influencia as decisões. As regras são uma simulação didática, não um modelo agronômico validado.

**Visual:** renderização da horta e uma lista curta de estado, ações e recompensa.

## 1:10–2:30 — Marcelo: experimentos e resultados

Comparamos DQN, PPO e A2C. Testamos quatro configurações por algoritmo, cada uma com três sementes de treinamento. Escolhemos a melhor pela média dos resultados de validação.

Depois, treinamos novamente as configurações escolhidas com cinco sementes novas. Cada modelo foi avaliado em cem episódios separados, sem continuar aprendendo. Também comparamos com ações aleatórias e uma regra simples que irriga o canteiro com maior necessidade.

Entre os algoritmos de aprendizado, {best.algorithm} teve o maior retorno médio: aproximadamente {best.episode_return_mean:.1f}, com desvio de {best.episode_return_std:.1f} entre treinamentos. Seu sucesso médio foi de {100 * best.success_mean:.1f} por cento. Sucesso significa terminar os sessenta turnos com todas as plantas vivas e saúde média de pelo menos cinquenta por cento.

A heurística teve retorno {heuristic.episode_return_mean:.1f}, e a política aleatória, {random.episode_return_mean:.1f}. O gráfico mostra por que é importante comparar várias sementes e olhar também saúde e água: gastar pouco pode significar simplesmente deixar plantas morrerem.

**Visual:** gráfico `reports/figures/comparacao.png`, mais uma execução curta do MP4 local. Valores completos para consultar: {ranking}.

## 2:30–2:50 — conclusão dividida entre integrantes

**Nicolas:** Conseguimos implementar o MDP e comparar três algoritmos com parâmetros e avaliações registrados.

**Marcelo:** Os resultados valem para este simulador. Como próximos passos, podemos usar sensores com ruído, outras plantas e maior busca de hiperparâmetros.

## Publicação

- Gravar fala real de todos os integrantes e conferir duração máxima de três minutos.
- Publicar no YouTube; o MP4 ilustrativo não substitui a apresentação.
- Incluir o link em `configs/entrega.json` e gerar o relatório novamente.
- Divulgação em rede social é opcional; inserir o link se realizada.
"""
        (ROOT / "apresentacao/roteiro.md").write_text(script, encoding="utf-8")
    print("Notebook executado e relatório HTML exportado.")


if __name__ == "__main__":
    main()

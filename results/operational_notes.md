# Registro de problemas operacionais

1. O piloto de diagnóstico encontrou mutação de `policy_kwargs` pelo A2C, causando falha ao serializar uma classe de otimizador. Foi corrigido entregando cópia profunda ao construtor. Os dados foram preservados em `pilot_diagnostico`; o piloto oficial foi refeito.
2. A fonte inicial da renderização não continha todos os acentos. A renderização passou a utilizar DejaVu Sans distribuída pelo Matplotlib, antes do piloto oficial e da busca.
3. O Windows exigiu execução fora do sandbox para o kernel local do Jupyter criar suas permissões de conexão e para criar canais entre processos. Essas permissões foram solicitadas pelas ferramentas; arquivos temporários permanecem na pasta local ignorada.
4. Após concluir os cinco treinamentos finais do DQN, o agendador tentou redefinir threads interop do PyTorch em um processo reutilizado. O PyTorch só permite definir esse parâmetro uma vez. A configuração passou para o inicializador de cada processo. Não houve alteração de ambiente, algoritmo, hiperparâmetros ou orçamento. Os resultados do DQN foram preservados e retomados sem repetição; os próximos treinamentos ainda não haviam sido iniciados.

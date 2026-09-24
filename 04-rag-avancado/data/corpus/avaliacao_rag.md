# Avaliação de sistemas de RAG

Um sistema de RAG tem dois componentes que falham por razões diferentes, e avaliá-los juntos esconde a causa do erro. Se a resposta final está errada, pode ser porque a passagem certa não foi recuperada, ou porque foi recuperada e o gerador não a usou. As duas situações exigem correções opostas.

## Métricas de recuperação

Estas métricas ignoram o gerador e medem apenas se os documentos certos chegaram ao topo.

Hit@k, também chamado de acurácia top-k, mede se ao menos um documento relevante aparece entre os k primeiros. É a métrica usada no DPR, onde o BM25 atinge 42,9% e o DPR 65,2% em top-5.

Recall@k é a fração dos documentos relevantes que foram recuperados entre os k primeiros. Precision@k é a fração dos k recuperados que são relevantes.

MRR, ou Mean Reciprocal Rank, é a média de 1 dividido pela posição do primeiro acerto. Premia colocar o documento certo logo no topo, o que importa porque o gerador sofre com o efeito Lost in the Middle.

nDCG@k, ganho cumulativo descontado normalizado, desconta a contribuição de cada acerto por um fator logarítmico da posição, e normaliza pelo ranking perfeito. É a métrica principal reportada no benchmark BEIR, geralmente com k igual a 10.

## Métricas de geração

Fidelidade, ou faithfulness, mede se cada afirmação da resposta é sustentada pelo contexto recuperado. Corresponde diretamente ao token IsSUP do Self-RAG.

Relevância da resposta mede se a resposta de fato endereça a pergunta feita, e não um assunto vizinho.

Acurácia de citação mede se as referências apontadas sustentam mesmo as afirmações que acompanham. O Self-RAG reporta ganhos expressivos justamente nessa métrica para gerações longas.

Taxa de abstenção mede com que frequência o sistema admite não ter informação suficiente. Uma taxa próxima de zero em um conjunto que contém perguntas sem resposta no corpus é sinal de alucinação, não de competência.

## Baselines indispensáveis

Duas comparações dão sentido a qualquer número reportado. O baseline closed-book responde sem nenhuma recuperação e mostra o ganho real que a recuperação trouxe. O baseline oracle fornece ao gerador a passagem correta de propósito e mostra o teto de desempenho — se o oracle já erra, o problema está no gerador ou no prompt, e melhorar o recuperador não vai adiantar.

## Perguntas sem resposta

Todo conjunto de avaliação deve incluir perguntas cuja resposta não está no corpus. Sem elas, não é possível distinguir um sistema que sabe responder de um sistema que sempre responde.

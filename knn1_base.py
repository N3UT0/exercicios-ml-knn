import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv ("churn_dataset.csv")

print(df.head())

x = df.drop('cancelou', axis=1)
y = df['cancelou']

# test.size=Quantidade de dados para teste, restante ate completar 1 vai para treino
# random_state=semente de aleatoridade, 42 é o valor comum da ocmunidade para garantir reprodutibilidade
# train_test_split=escolhe aleatoriamente os dados que vão para teste e para treino 

# x=dados de entrada
# y=rotulos de saida


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit (x_train, y_train)

y_pred=knn.predict(x_test)

accuracy = accuracy_score (y_test,y_pred)


k_values = [1,3,5,7,9]
results={}

for k in k_values:
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)
    preds=model.predict(x_test)
    acc=accuracy_score(y_test,preds)
    results[k]=acc
    print(f'Acuracia com k-{k}: {acc:.2}')

print('\n Resumo de Acuracia por K: ')



print('=-'*30)
print('O Valor de K com a melhor acuracia foi: 1')
print('O modelo errava mais quando o valor de K era maior')
print('O valor de K é oque define o quanto ele vai considerar os vizinhos, se colocar um valor baixo ele ira considerar menos os vizinhos, porem se tornara mais instavel, ao colocar um valor alto ele cosiderar mais os vizinhos, fazendo com que a acuracia seja menor porem mais confiavel e estavel')

print('\nPara resumir, o K alto ele considera a opinião da maioria por assim dizer, o K menor considera somente oque ele pensou')
#calculate the weighted sum

bias = 0.5
l_rate = 0.01
epochs = 30
epoch_loss = []


data, weights = generate_data(50,3)

def train_model(data, weights, bias, l_rate, epochs):
  for e in range(epochs):
    individual_loss = []

    for i in range(len(data)):
      feature = data.loc[i][:-1].to_numpy()
      target = data.loc[i][-1]
      w_sum =  generate_weighted_sum(feature, weights, bias)
      prediction = sigmoid(w_sum)
      loss = cross_entropy_loss(target, prediction)
      individual_loss.append(loss)
      #GRADIENT DESCENT

      weights = update_weights(weights, l_rate, target, prediction, feature)
      bias = update_bias(bias, l_rate, target, prediction)

    average_loss = sum(individual_loss)/ len(individual_loss)
    epoch_loss.append(average_loss)
    print('************************')
    print('epoch', e)
    print(average_loss)


def generate_weighted_sum(feature, weights, bias):
  return np.dot(feature, weights) + bias

def sigmoid(w_sum):
  return 1/(1+np.exp(-w_sum))

def cross_entropy_loss(target, prediction):
  return -(target * np.log10(prediction) + (1-target) * np.log10(1-prediction))

def update_weights(weights, l_rate, target, prediction, feature):
  new_weights = []
  for x,w in zip(feature, weights):
    new_w = w + l_rate*(target-prediction)*x
    new_weights.append(new_w)
  return new_weights

def update_bias(bias, l_rate, target, prediction):
  return bias + l_rate*(target - prediction)

train_model(data, weights, bias, l_rate, epochs)

#PLOT THE AVERAGE LOSS CURVE

df = pd.DataFrame(epoch_loss)
df_plot = df.plot(kind='line', grid=True).get_figure()
df_plot.savefig('Training_Loss.pdf')


# tensorflow

* TensorBoard: Visualizing Learning
 
- 텐서플로우 그래프를 만든다. 그리고 어떤 노드들을 summary operation 들을 이용하여 표시할것인지를 결정한다
    - 예를 들어 만약 MNIST digits 을 확인하는 나선형의 신경망을 처리한다고 하자. 아마도 당신은 learning rate가 시간이 지나면서 변하거나, objective function 이 변하는 것을 기록할 것이다. scalar_summary ops 를 노드들에 붙이면 이것들을 기록한다. 그런다음 각각의 scalar_summary에 'learning rate' 나 'loss function' 등 의미있는 태그를 붙여주면 된다.
    - 정리
        1. 목표를 세움.
        2. 목표에 필요한 데이터 콜렉팅 
        3. 알맞은 summary ops 를 설정
        4. 설정된 summary에 알맞은 tag 설정
        5. 끝.
    
    - summary 에 대한 자세한 사항은 [링크 클릭](https://www.tensorflow.org/api_docs/python/summary/)

    - op 라는 개념을 잘 이해를 못하겠음. 

    - 텐서플로우의 작업들은 그것들을 실행하기 전까지 아무것도 하지 않는다. 그리고 위에서 설명한 summary 노드들도 너의 그래프에 따른 주변요소이다. 그러므로 summary 를 만들기 위해서는 우리는 모든 summary 노드들을 실행해야한다. 이것들을 일일이 하는것은 매우 지겹기 때문에 tf.summary.merge_all 을 사용해라. 

    
    - 그다음에는 merged 된 summary op 를 실행하면, 모든 summary 데이터가 담겨진 serialized된 Summary protobuf object이 만들어진다. 이 summary 데이터를 디스크에 저장하기 위해서는 **tf.train.SummaryWriter** 에 이 데이터를 넘기면 된다. 

    - 정리
        1. 텐서플로우는 실행하기 전까지 아무것도 않함
        2. summary 를 그래프의 주변요소
        3. 모든 summary 를 실행하려면 tf.summary.merge_all 사용 
        4. 3번에 summary protobuf object 이 만들어짐 
        5. 이걸 tf.train.SummaryWriter 에 넘기면 디스크 저장할수 있음
        6. 끝.

    - SummaryWriter 는 logdir 을 생성자에 파라미터로 받는다. 이 폴더가 모든 이벤츠가 저장되는 공간이다. (중요) 그리고 SummaryWriter 는 가끔씩 graph 를 생성자에서 파라미터로 받는다. graph 공간이 주어진다면, TensorBoard 는 너의 그래프를 보여줄것이다. 그래프로 보는게 너한테 훨씬 좋을걸? 

    - 너가 수정된 그래프와 SummaryWriter 가 있다면 너는 시작할 준비가 됐다! (따다~) 인제 밑에 예제 보고 한번 따라해보렴. [여기](https://www.tensorflow.org/tutorials/mnist/beginners/)에 설명서 있고 [기요](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)에 코드있으니 참고하고.. 

~~~
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  dropped = tf.nn.dropout(hidden1, keep_prob)

# Do not apply softmax activation yet, see below.
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
  # The raw formulation of cross-entropy,
  #
  # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                               reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the
  # raw outputs of the nn_layer above, and then average across
  # the batch.
  diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                      sess.graph)
test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
tf.global_variables_initializer().run()
~~~

- 모든 SummaryWriters 들을 initialize 한다음, summary 를 붙여서 test 하면 됨. 

~~~
# Train the model, and also write summaries.
# Every 10th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data, & add training summaries

def feed_dict(train):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train or FLAGS.fake_data:
    xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
    k = FLAGS.dropout
  else:
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

for i in range(FLAGS.max_steps):
  if i % 10 == 0:  # Record summaries and test-set accuracy
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # Record train set summaries, and train
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)
~~~

- 인제 모든 준비 끝!


# 텐서보드 구동하기
- 텐서보드 시작하려면,
~~~
tensorboard --logdir=path/to/log-directory
~~~

- 여기서 logdir 은 SummaryWriter 가 데이터를 serialize 해 둔 곳임. 
- 텐서보드가 실행됐다면, 인제 브라우저에서 localhost:6006 가서 확인 ㄱㄱㄱ
- graph 에 대한 더 많은 정보는 [요기서](https://www.tensorflow.org/how_tos/graph_viz/)
- 텐서보드 사용에 대한 더 많은 정보는 [요기로](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/README.md) 








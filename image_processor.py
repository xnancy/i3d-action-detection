import tensorflow as tf 


def image_printer(filepaths): 
	filename_queue = tf.train.string_input_producer(filepaths) #  list of files to read

	reader = tf.WholeFileReader()

	key, value = reader.read(filename_queue)

	image = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.


	# Start a new session to show example output.
	with tf.Session() as sess:

	    # Coordinate the loading of image files.
	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(coord=coord)

	    # Get an image tensor and print its value.
	    image_tensor = sess.run(image)
	    print(image_tensor[0][0][0])

	    # Finish off the filename queue coordinator.
	    coord.request_stop()
	    coord.join(threads)

def train_image_processor(filepaths): 

	filename_queue = tf.train.string_input_producer(filepaths) #  list of files to read

	reader = tf.WholeFileReader()

	key, value = reader.read(filename_queue)

	image = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.


	# Start a new session to show example output.
	with tf.Session() as sess:

	    # Coordinate the loading of image files.
	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(coord=coord)

	    # Get an image tensor and print its value
	    shape = sess.run(tf.shape(image))
	    # Should be ~100 for Something-Something
	    min_dimension = min(shape[0], shape[1])
	    print min_dimension
	    print shape

	    # Finish off the filename queue coordinator.
	    coord.request_stop()
	    coord.join(threads)


filepaths = ['Data/00001.jpg'] 
train_image_processor(filepaths)

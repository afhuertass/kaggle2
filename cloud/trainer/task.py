import tensorflow as tf
from tensorflow.python.lib.io import file_io
import cPickle
import numpy as np
import sys


import dnc
import input_manager
# 49688
OUTPUT_SIZE = 49690
BATCH_SIZE =  40

# Local parameters
"""
PATH_PRODUCTS = "../data/csvs/products.csv"
PATH_TRAIN_DATA = [ "../data/train-2.pb2" ]
PATH_TEST_DATA = ["../data/train-2.pb2" ]
CHECK_DIR = "../checkpoints"
TB_DIR = "../tensorboard"
"""
### Cloud parameters


PATH_TRAIN_DATA = [ "gs://kaggleun-instacart/data/train.pb2" , "gs://kaggleun-instacart/data/train2.pb2" ]
PATH_TEST_DATA = ["gs://kaggleun-instacart/test.pb2" ]
PATH_PRODUCTS = "gs://kaggleun-instacart/data/products/products.csv"
CHECK_DIR = "gs://kaggle_instacart_model"
TB_DIR = "gs://kaggle_instacart_tb"


# total train objects = 50000
n = 100 # actual number of runnings over all the training data 
NUM_ITER = 3
NUM_ITER_TEST = 10 
NUM_ITER = (50000/BATCH_SIZE)*n  # numero the training epochs 
NUM_ITER_TEST = (75000/BATCH_SIZE)*1 # para obtener las prediciones

#test delete for training
REP_INTERVAL = 100

NUM_ITER_TEST = 10

MAX_GRAD_NORM = 50
LEARN_RATE = 1e-3
MULTIPLIER = 0.1
reduce_learning_interval = 1000
EPSILON = 1e-3


#CHECK_DIR = "gs://kaggle_instacart_model"
# TB_DIR = "gs://kaggle_instacart_tb"
CHECK_INTERVAL = 1000



access_config = {
        'memory_size': 128 ,
        'num_writes' : 1 ,
        'num_reads' : 2  ,
        'w_size' : 64 
    }
controller_config = {
        'num_hidden'  : 8 ,
        'depth' : 2 ,
        'output_size' : OUTPUT_SIZE
    }


def run_model2( dnc_core , initial_state  , inputs_sequence , seqlen  , output_size ):


    output_sequence , _ = tf.nn.dynamic_rnn(
        cell = dnc_core ,
        inputs = inputs_sequence ,
        sequence_length= seqlen,
        time_major = True ,
        initial_state = initial_state 
    )

    return output_sequence 

def train( num_epochs , rep_interval):

    
    ## create the dnc_core 
    dnc_core = dnc.DNC( access_config = access_config , controller_config  = controller_config , output_size = OUTPUT_SIZE )
    
    initial_state = dnc_core.initial_state(BATCH_SIZE)

    #load the data
    input_data = input_manager.DataInstacart( PATH_PRODUCTS, BATCH_SIZE  )
    
    input_tensors = input_data(PATH_TRAIN_DATA , num_epochs )

    # load the test data
    input_tensors_test = input_data(PATH_TEST_DATA , NUM_ITER_TEST ) # una sola pasada 
    
    output_sequence = run_model2( dnc_core , initial_state , input_tensors[0] , input_tensors[3] , OUTPUT_SIZE  )

    output_sequence_test = run_model2(  dnc_core , initial_state , input_tensors_test[0] , input_tensors_test[3] , OUTPUT_SIZE )
    # last output from the recurrent neural network 
    last_rnn = tf.gather( output_sequence , int( output_sequence.get_shape()[0] - 1  )  )

    last_rnn_test = tf.gather( output_sequence_test , int( output_sequence_test.get_shape()[0] - 1  )  )
    
    print("wash going on two")
    print(last_rnn.shape)
    print( input_tensors[1])
    train_loss = input_data.cost(  last_rnn , input_tensors[1] )

    #eval_loss = input_data.cost_f1( last_rnn , input_tensors[1] )
    
    
    tf.summary.scalar( 'loss' , train_loss  )
    
    trainable_variables = tf.trainable_variables()

    grads , _ = tf.clip_by_global_norm(
        tf.gradients( train_loss , trainable_variables) , MAX_GRAD_NORM
    )

    learning_rate = tf.get_variable(
        "learning_rate" , shape = [],
        dtype = tf.float32 , initializer = tf.constant_initializer(LEARN_RATE) ,
        trainable = False 
    )
    
    reduce_learning_rate = learning_rate.assign( learning_rate*MULTIPLIER  )
    
    global_step = tf.get_variable(
        name="global_step" ,
        shape = []  ,
        dtype = tf.int64 ,
        initializer = tf.zeros_initializer() ,
        trainable = False ,
        collections = [ tf.GraphKeys.GLOBAL_VARIABLES , tf.GraphKeys.GLOBAL_STEP]
    )

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate , epsilon = EPSILON
    )

    train_step = optimizer.apply_gradients(
        zip(grads, trainable_variables) , global_step = global_step
    )


    
    saver = tf.train.Saver()
    tf.summary.scalar( 'loss' , train_loss  )

    merged_op = tf.summary.merge_all()

   
    
    if CHECK_INTERVAL > 0:

        hooks = [
            tf.train.CheckpointSaverHook(
                checkpoint_dir = CHECK_DIR ,
                save_steps = CHECK_INTERVAL ,
                saver = saver 
            )
        ]
    else:
        hooks = []

   
    
    with tf.train.SingularMonitoredSession( hooks = hooks , checkpoint_dir = CHECK_DIR ) as sess:

        writer = tf.summary.FileWriter( TB_DIR , sess.graph )
        
        start_iteration = sess.run(global_step)
        total_loss = 0

        print("start:{}".format( start_iteration ) )
        
        for train_iteration in xrange(start_iteration , num_epochs):

            #t =  sess.run( input_tensors[0] ) # feats
            # 
            break 
            _ , loss = sess.run( [ train_step , train_loss] )
            
            if train_iteration % 100 == 0 :
                summary  = sess.run( merged_op  )
                writer.add_summary(summary , train_iteration )
            

            if ( train_iteration  + 1 )% reduce_learning_interval == 0:
                sess.run( reduce_learning_rate )
                print("reducing learning rate")
                
            if train_iteration % 500 == 0 :
               
                print( "loss:{}".format(loss)  )
               
                print( "step-training:{}".format( train_iteration ) )
            
           

    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    result = "order_id,products\n"

    L = [] 
    with tf.Session() as sess:
        sess.run( init_op )
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess , coord = coord )
        try:
            step = 0
            for i in xrange(0 , NUM_ITER_TEST):
            #while not coord.should_stop():

                if i % 1000 == 0:
                    print( "step:{}".format( step ) )
                    # retrieve prediction , y el id
                    
                prediction , idd  = sess.run( [last_rnn_test , input_tensors_test[2] ] )
                L.append( ( prediction,idd)  )
                
                human = input_data.to_human_read( prediction , idd   )
                for e in human:
                    result += e 
                    #predfile.write(e)
                step = step + 1
                
        except tf.errors.OutOfRangeError :
            print("Exhausted Queue ")
        finally:
            
            coord.request_stop()
            coord.join(threads)
            sess.close()

    with file_io.FileIO(CHECK_DIR+"/subme-n.pickle" , 'w+') as f:

        cPickle.dump( result , f )
        print("printed")

    with file_io.FileIO(CHECK_DIR+"/full-result.pickle" , 'w+') as f:
        cPickle.dump( L , f )
        print("pickled")
        
    
def main( unuser_args):

    train(NUM_ITER , REP_INTERVAL)
    
    print("riko -train ")

if __name__=="__main__":

    tf.app.run()

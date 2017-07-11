
import multiprocessing

import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

import sonnet as snt

import pandas as pd 
import numpy as np 
# this class will handle the inputs. 

# this is for the onehot encoding vector

TOTAL_ITEMS = 49690 # total podructs
LEN = 150  #
LEN_MAX = 150
def parse_examples( examples ):
        # parse the data from the file, to an apropiated format
    
    feature_map = {
        'target': tf.VarLenFeature(
            dtype = tf.int64
        ) ,
        'feature': tf.VarLenFeature(
        dtype = tf.int64
        ) ,

       'ids' : tf.VarLenFeature(
           dtype = tf.int64
       )
    }
    parsed = tf.parse_example(examples , feature_map )

    print("expliquese gonorrea")
    
    
    idd =  tf.reshape( parsed['ids'].values[:1] , shape = [1] )

   
    """
    so far here we have feature = [ n_inputs ] , 
    target = [ n_targets ] ,  targets should be a one hot encoding
    
    """
    print("shape target")
   

    sparse_target_indices = tf.reshape( parsed['target'].values[:] , shape=[ -1] )
    print("shape target indiciiiii")
    print(sparse_target_indices.shape)
   
    
    target = tf.one_hot( sparse_target_indices , TOTAL_ITEMS , on_value = 1.0 ,
                         off_value = 0.   )

    print(target.shape )
    
    #target = tf.one_hot( parsed['target'] , TOTAL_ITEMS , on_value = 1.0 ,
                         #off_value = 0. )
    # target [ n_target , TOTAL_ITEMS ]
    
    print("shape target reduced")
    target = tf.reduce_sum( target , 0   )
    
    target = tf.reshape( target , [ -1 ,TOTAL_ITEMS ] )
    print(target.shape) 
    print("targets good")

    print("asdasdasdadas")


    
    
    sparse_feature_indices = tf.reshape(  parsed['feature'].values[:LEN] , shape=[ LEN ]  )
    # len
    seqlen = tf.count_nonzero( sparse_feature_indices )
    
    # contar el 
    
    #sparse_feature_indices = tf.to_int64( sparse_feature_indices )
    print("sparse features aaaa")
    print( sparse_feature_indices.shape )
    features = tf.one_hot( sparse_feature_indices , TOTAL_ITEMS , on_value = 1.0 ,
                           off_value = 0.0 ,  )
    
    print( features.shape )
    features = tf.reshape( features , shape = [   LEN  ,TOTAL_ITEMS ]  )
    
    print("one hot re hot ")
        
    # target [ TOTAL_ITEMS]
    # features [ n_inputs , TOTAL_ITEMS ] total 

    print("shapes FINAL")
    print( features.shape )
    print( target.shape )
    


    
    return features , target , idd , seqlen 
    

class DataInstacart(snt.AbstractModule):

    def __init__(self ,  path_products , batch_size   , name = 'data_m'):


        super(DataInstacart , self ).__init__(name = name )

        self.batch_size = batch_size

        
        
        self.shape_sample = [ LEN , -1 , TOTAL_ITEMS  ]
        self.shape_target = [  -1 , TOTAL_ITEMS ]
        self.shape_id = [ -1  , 1 ] 
        

        #self.df_products_c = pd.read_csv( path_products , dtype ={ 'product_id': int , 'product_name' : str , 'aisle_id': int , 'deparment_id': int  } )

        #self.df_products = self.df_products_c['product_id']
        self.df_products = np.arange( 1 , 49690)
        # lista de productos 
            
    def _build(self, data_files , num_epochs ):
        # recieves the data files where the .tfrecord lies.
        # so it build the queue to read from the files and retrieve the data tensors
        
       
        # build the shit
        print(data_files)
        thread_count = multiprocessing.cpu_count()
        min_after_dequeue = 10
        queue_size_multiplier = thread_count + 3
        capacity = self.batch_size*2
        
        filename_queue = tf.train.string_input_producer(   data_files , num_epochs = num_epochs   )
        
        _ , encoded_examples = tf.TFRecordReader(
            options = tf.python_io.TFRecordOptions  (
                compression_type= TFRecordCompressionType.GZIP
            )
        ).read_up_to( filename_queue , self.batch_size)


        #features, target = parse_examples( encoded_examples )
        feature, target , idd , seqlen =  parse_examples( encoded_examples )
        # define certain capacity
        
        
        print("shapes train batchs")
        print( feature.shape )
        print( target.shape  )
        
        # of change to a shuffle batch 
        result = tf.train.shuffle_batch(
            [feature,target , idd  , seqlen ] ,
            batch_size = self.batch_size  ,
            capacity = capacity ,
            min_after_dequeue =  min_after_dequeue
        )

        print("waaaaaaaaaaaaaaaaaa")
        print( self.batch_size )
        print(result[0].shape)
        print(result[1].shape)
        print(result[2].shape)
        
        result[0] = tf.reshape( result[0] , self.shape_sample )
        result[1] = tf.reshape(  result[1] , self.shape_target )
        result[2] = tf.reshape( result[2] , [-1 , 1 ])
        
        #print( result )
        # feature [ LEN , batch_size , TOTAL ]
        # target  [ batch_size ,  TOTAL ]

        print("shapes train batchs FINAL")
        print( result[0].shape )
        print( result[1].shape  )

        
        return result  

    def to_human_read(self , data , ids ):
        # ids [ batch_size ] 
        # data [ batch , LEN ]

        # return [batch , "string" ]
        batch_size = data.shape[0]
        vec_size = data.shape[1]
        indx = 0
        print("adsadasdas")
        for idd, dat  in  zip( ids , data) :
            
            index = np.where( dat > 0.3 )[0]
            print("some info")
            print( dat.shape )
            print( np.max( dat ) )
            print( index ) 
            if len(index) == 0 :
                print("no index")
                yield "{},{}\n".format(idd[0], "None")
            else:
                resp = ""
                for el in index:
                    if el == 0 :
                        continue
                    resp += str(el)
                    resp += " "
                    #print( "{} , {}".format( idd[0] , resp )   )
                yield "{},{} \n".format( idd[0] , resp )
            
        
    def cost( self , last_output , target ):
    # last output [ batch_size , ToT ]
    # target [ batch_size , TOT ]

    # sigmoid cross entropy, because the classes arent mutially exclusive

        
        xent = tf.nn.sigmoid_cross_entropy_with_logits( logits = last_output , labels = target )
        xent_batch = tf.reduce_mean( xent , axis = 1)

        batch_size = tf.cast(tf.shape(last_output)[0], dtype=xent_batch.dtype)
        
        xent_total = tf.reduce_sum( xent_batch ) / batch_size
        
        print(  xent_total.shape ) 
        return  xent_total
    

    def cost_f1( self , last_output , target ):

        TP = tf.count_nonzero( target*last_output )
        TN = tf.count_nonzero( (last_output -1 )*(target -1 ) )
        FP = tf.count_nonzero( last_output*(target-1) )
        FN = tf.count_nonzero( (last_output-1)*target )

        precision = TP/(TP + FP )
        recall = TP/( TP + FN )

        f1mean_batch = tf.reduce_sum(   2*precision*recall/(  precision + recall ) , axis = 1  )
        
        batch_size = tf.cast(tf.shape(last_output)[0], dtype=xent_batch.dtype)
        
        return   tf.reduce_sum( f1mean_batch)/batch_size
    
    

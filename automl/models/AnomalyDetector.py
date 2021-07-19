import ray
import numpy as np
import pandas as pd
import FraxsesModel
import tensorflow as tf 
from inspect import signature
from utils.local_settings import VERBOSE

@ray.remote
class AnomalyDetector(FraxsesModel):

    def __init__(self, config:dict):
        super().__init__(config=config)
        try:
            self.index_schema = config['index_schema']
            self.context_schema = config['context_schema']
            self.target_schema = config['target_schema']
            self.target_col = config['target_schema'][0]['column_name']
        except KeyError as e:
            self.logger.error(f'Invalid Configuration: {e}')
        else:
            self.window = config.get('window', 12)
            self.epochs = config.get('epochs', 10)
            self.dropout = config.get('dropout', 0.2)
            self.batch_size = config.get('batch_size', 32)
            self.optimizer = config.get('optimizer', 'Adam')
            self.momentum = config.get('momentum', None)
            self.learning_rate = config.get('learning_rate', 0.001)
            self.activation_fn = config.get('activation_fn', 'relu')
            self.loss_fn = config.get('loss_fn', 'mse')
            self.bookmark = self.df.iloc[0].name
            # self._tune_config = {
            #     'epochs': tune.randint(1, 1000),
            #     'dropout': tune.uniform(0.1, 0.9),
            #     'batch_size': tune.grid_search([32, 64, 128]),
            #     'optimizer': tune.choice(['adam', 'sgd']),
            #     'learning_rate': tune.uniform(0.001, 0.1),
            #     'momentum': tune.uniform(0.1, 0.9),
            #     'activation_fn': tune.choice(['relu', 'tanh']),
            #     'loss_fn': tune.choice(['mse']),
            #     'window': tune.randint(7, 1000),
            # }

    def build_model(self):
        '''Builds a Tensorflow convolution autoencoder'''
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=(self.window, self.num_features)
                ),
                tf.keras.layers.Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2, activation=self.activation_fn
                ),
                tf.keras.layers.Dropout(rate=self.dropout),
                tf.keras.layers.Conv1D( 
                    filters=16, kernel_size=7, padding="same", strides=2, activation=self.activation_fn
                ),
                tf.keras.layers.Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2, activation=self.activation_fn
                ),
                tf.keras.layers.Dropout(rate=self.dropout),
                tf.keras.layers.Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2, activation=self.activation_fn
                ),
                tf.keras.layers.Conv1DTranspose(
                    filters=1, kernel_size=7, padding="same"
                ),
            ]
        )
        
        self.model.summary(print_fn=self.logger.debug)

    @ray.method(num_returns=2)
    def format_data(self, mode:str='train'):
        '''
        Format the dataframe for time series anomaly detection

        Args: 

        mode (str): options are "train" or "predict"

        Returns:

        dataset: dataset containg numpy array(s)
        '''

        CV_SPLITS = 5 
        SYNTHETIC_ANOMALY_FREQUENCY = 0.05
        
        df = self.df # .to_koalas()      
        index_col_nm, index_col_dtype, index_col_format = map(list, zip(*[(col_obj['column_name'], col_obj['dtype'], col_obj['format']) for col_obj in self.index_schema]))
        context_col_nm, context_col_dtype, context_col_format = map(list, zip(*[(col_obj['column_name'], col_obj['dtype'], col_obj['format']) for col_obj in self.context_schema]))
        target_col_nm, target_col_dtype, target_col_format = map(list, zip(*[(col_obj['column_name'], col_obj['dtype'], col_obj['format']) for col_obj in self.target_schema]))

        cols = context_col_nm + target_col_nm + index_col_nm
        dtypes = context_col_dtype + target_col_dtype + index_col_dtype
        formats = context_col_format + target_col_format + index_col_format

        # drop unused columns
        dropcols = [c for c in df.columns if c not in cols]
        self.logger.debug(f'dropping columns: {dropcols}')
        df = df.drop(columns=dropcols)
        self.logger.debug(f'remaining columns: {list(df.columns)}')

        # enforce schema
        self.logger.debug('enforcing schema')
        for c,d,f in zip(cols, dtypes, formats):
            if d == 'datetime': 
                df[c] = pd.to_datetime(df[c], format=f if f else None)
            else:
                df[c] = df[c].astype(np.dtype(d))
        
        # set index
        self.logger.debug(f'setting index: {index_col_nm[0]}')
        df = df.set_index(index_col_nm[0])
        
        # engineer features
        self.logger.debug('engineering features')
        categorical = []
        numeric = []
        
        for i,c in enumerate(cols):
            if formats[i] == 'numeric':
                numeric.append(c)
            elif formats[i] == 'categorical':
                categorical.append(c)

        # one hot encode the categorical data
        self.logger.debug(f'one hot encoding columns: {categorical}')
        if categorical:
            df = pd.get_dummies(df, columns=categorical, prefix=categorical) 

        # peek at dataframe before creating datasets
        self.logger.debug(
            f'df memory usage: {df.memory_usage(deep=True).sum()} bytes' + \
            f'\n{df.head()}'
        )

        # format data based on mode
        self.logger.debug(f'formating data mode: {mode}')

        if mode=='train':
            sequences = []
            total_sequences = len(df) - self.window + 1
            # TODO: make insertion more random
            anomaly_insertion_count = int(len(df) * SYNTHETIC_ANOMALY_FREQUENCY)
            anomaly_insertion_period = int(np.floor(total_sequences / anomaly_insertion_count))
            anomaly_idx = 0

            self.logger.debug(
                f'\ntarget_col_nm : {target_col_nm}\n' + \
                f'total_sequences: {total_sequences}\n' + \
                f'anomaly_insertion_count: {anomaly_insertion_count}\n' + \
                f'anomaly_insertion_period: {anomaly_insertion_period}\n' + \
                f'num_features: {df.shape[1]}'
            )

            self.logger.debug('setting num_features class attribute')
            self.num_features = df.shape[1]
            
            # create sequences 
            target_idx = [df.columns.get_loc(c) for c in target_col_nm if c in df]
            test = True
            for i in range(0, total_sequences):
                sequence = df.iloc[i : (i + self.window)].values
                anomaly_labels = np.zeros((self.window, 1))
                
                # add random synthetic anomaly, used std rule for magnitude      
                if i > 0 and i % anomaly_insertion_period == 0:
                    # choose a random index for the anomaly
                    anomaly_idx = np.random.randint(0, len(sequence))

                    # make the anomaly an outlier value
                    quartiles = np.random.choice([-3,3])
                    z = np.ceil(sequence[target_idx].std() % 10) * np.random.randint(-5,5)
                    anomaly_value = sequence[target_idx].mean() + quartiles*sequence[target_idx].std() + z

                    # insert the anomaly into the sequence
                    sequence[target_idx, anomaly_idx] += anomaly_value
                    if test:
                        self.logger.debug(f'sample sequence w/synthetic anomaly:\n{sequence[target_idx]}')
                        test = False

                    # add the current index to the anomaly index list
                    anomaly_labels[anomaly_idx] = 1
                elif anomaly_idx:
                    # adjuist the anomaly index to insert it into the next sequence if within the sequence range
                    anomaly_idx -= 1

                    # insert the anomaly into the sequence
                    sequence[target_idx, anomaly_idx] += anomaly_value
                    anomaly_labels[anomaly_idx] = 1

                # add anomaly label column to the sequence then add to sequences list
                sequences.append(np.hstack((sequence, anomaly_labels)))

            # split data into cross-validation datasets
            train = []
            test = []
            splits = list(map(np.stack, np.array_split(sequences, CV_SPLITS)))
    
            for i, split in enumerate(splits):
                self.logger.debug(f'split {i} shape:{split.shape}')

            for i in range(0, 3):
                tr = np.concatenate(splits[0:2+i])
                te = splits[2+i]
                
                # normalize the numeric data
                if numeric:
                    numeric_idx = [df.columns.get_loc(c) for c in numeric if c in df]
                    
                    # normalize training data independently
                    tr_mean = tr[numeric_idx].mean()
                    tr_std = tr[numeric_idx].std()
                    tr[numeric_idx] = (tr[numeric_idx] - tr_mean) / tr_std

                    # normalize testing data independently
                    te_mean = te[numeric_idx].mean()
                    te_std = te[numeric_idx].std()
                    te[numeric_idx] = (te[numeric_idx] - te_mean) / te_std
            
                train.append(tr)
                test.append(te) 
            
            # Set bookmark as last index in training data set
            self.logger.debug(f'setting bookmark: {self.df.iloc[train[-1].shape[0]].name}')
            self.bookmark = self.df.iloc[train[-1].shape[0]].name

            return train, test

        elif mode=='predict': 
             # normalize the numeric data
            if numeric:
                numeric_idx = [df.columns.get_loc(c) for c in numeric if c in df]
                mean = df[numeric_idx].mean()
                std = df[numeric_idx].std()
                df[numeric_idx] = (df[numeric_idx] - mean) / std
            
            # create sequences 
            sequences = []
            total_sequences = len(df) - self.window + 1
            
            for i in range(0, total_sequences):
                sequence = df.iloc[i : (i + self.window)]
                sequences.append(sequence.values)

            return np.array(sequences), None

    def lr_schedule(self, epoch:int, lr:float):
        '''Schedule for the learning rate'''
        if epoch < self.epochs/2:
            return lr*tf.math.exp(-0.1)
        else:
            return lr*tf.math.exp(-0.01)

    def train(self):
        '''Trains the model using the current configuration'''

        #### Ray something note
        # As a rule of thumb, the execution time of step should 
        # be large enough to avoid overheads (i.e. more than a few seconds), 
        # but short enough to report progress periodically (i.e. at most a few minutes).
        
        train_splits, test_splits = self.format_data(mode='train')

        callbacks = []

        if self.version:
            self.logger.debug(f'updating existing model to version {self.version + 1}...')
            # NOTE: https://arxiv.org/abs/2002.11770
            # TODO: momentum and learning rate selection
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(self.lr_schedule, verbose=int(VERBOSE)))
        else:
            self.logger.debug(f'training new model ...')
            self.build_model()

        opt_class = eval(f'tf.keras.optimizers.{self.optimizer}')

        if self.momentum and 'momentum' in signature(opt_class).keys():
            opt = opt_class(learning_rate=self.learning_rate, momentum=self.momentum)
        else:
            opt =  opt_class(learning_rate=self.learning_rate)    

        self.model.compile(
            optimizer=opt,
            loss=self.loss_fn,
        )

        # cross validate
        scores = []
        thresholds = []
        self.logger.debug(f'model.input_shape: {self.model.input_shape}')
        for i, (train, test) in enumerate(zip(train_splits, test_splits)):
            # remove anomaly labels
            tr = train[:,:,:-1]
            te = test[:,:,:-1]
            self.logger.debug(f'cross validation {i} training dataset shape: {tr.shape}')
            self.logger.debug(f'cross validation {i} testing dataset shape: {te.shape}')

            # train model
            history = self.model.fit(
                tr,
                tr, 
                batch_size=self.batch_size,
                epochs=int(np.ceil(self.epochs/2)) if self.version else self.epochs,
                validation_data=(te, te),
                callbacks=callbacks,
                verbose=VERBOSE,
            )
            
            # make predictions
            tr_p = self.model.predict(tr)
            te_p = self.model.predict(te)
            
            # calculate error
            tr_s = np.mean((tr_p - tr)**2)
            te_s = np.mean((te_p - te)**2)

            # define threshold based on training error
            threshold = np.mean(tr_s) + np.std(tr_s)*3
            thresholds.append(threshold)

            # calculate p_score (average reconstruction error) [capturing normal signal]
            r_score = np.mean(te_s)

            # calculate c_score (average classification error) [capturing anomalies]
            # p_anomalies = te_s > threshold
            # c_score = np.mean(tf.keras.losses.binary_crossentropy(test, p_anomalies))
            c_score = r_score
            
            # calculate score (average of classification and prediction error)
            score = (r_score + c_score)/2
            scores.append(score)

        # TODO: does this make sense?
        self.threshold = np.array(thresholds).mean()

        # report average score for all cross validations
        return np.mean(scores)

    def cleanup(self):
        '''Invoked when training is finished'''
        self.version += 1
        self.save_checkpoint() # TODO: will this cause confusion on latest version
        self.save_config(
            index_schema=self.index_schema,
            context_schema=self.context_schema,
            target_schema=self.target_schema,
            target_col=self.target_col,
            window=self.window,
            epochs=self.epochs,
            dropout=self.dropout,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            momentum=self.momentum,
            learning_rate=self.learning_rate,
            activation_fn=self.activation_fn,
            loss_fn=self.loss_fn,
            bookmark=self.bookmark,
            threshold=self.threshold,
            _dataset=self._dataset,
            _organization=self._organization,
            version=self.version,
            num_features=self.num_features,
        )

    def predict(self):
        '''Make predictions using the latest model'''
        samples, _ = self.format_data(mode='predict')
        preds = self.model.predict(samples, verbose=VERBOSE)
        error = np.mean((preds - samples)**2)
        anomalies = error > self.threshold 
        self.logger.debug("Number of anomaly samples: ", np.sum(anomalies))
        
        # TODO: verify below
        # anomalous_data_indices = []
        # for data_idx in range(self.window - 1, len(samples) - self.window + 1):
        #     if np.all(anomalies[data_idx - self.window + 1 : data_idx]):
        #         anomalous_data_indices.append(data_idx)
        # self.df['is_anomaly'] = 0
        # self.df['is_anomaly'].iloc[anomalous_data_indices] = 1
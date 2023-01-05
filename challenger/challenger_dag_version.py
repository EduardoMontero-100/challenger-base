import pandas as pd
import os
import pickle
from subprocess import PIPE, Popen
import pyspark.sql.functions as F
from pyspark.sql.types import ShortType
from pyspark.sql import DataFrame
import numpy as np
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import GBTClassifier 
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from pyspark.ml.tuning import CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV #Todo: Eliminar, No se usa
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score
import datetime;


class Challenger():
    def __init__(self,BALANCEAR_TARGET:bool, 
                 ELIMINAR_CORRELACIONES:bool, 
                 CASTEAR_BIGINT:bool, 
                 REDONDEAR_DECIMALES:bool, 
                 CON_SCALER:bool, 
                 TIENE_TESTING:bool, 
                 CORRER_RF:bool, 
                 CORRER_GB:bool, 
                 CORRER_LGBM:bool, 
                 CORRER_XGB:bool, 
                 CORRER_PRODUCTIVO:bool, 
                 CAMPO_CLAVE:str,
                 TARGET:str, 
                 modelo:str,
                 ABT_VARIABLES:str,
                 ABT_TABLA:str, 
                 TGT_TABLA:str, 
                 TGT_VARIABLES:str,
                 TGT_BALENCEO:int, 
                 DECIMALES_VARIABLES_NUMERICAS:int, 
                 COTA_CORRELACIONES:float, 
                 REGISTROS_X_PARTICION:float, 
                 PORCENTAJE_TRAINING:float, 
                 GB_param_test:dict, 
                 LGBM_param_test:dict, 
                 XGB_param_test:dict,
                 RF_param_test:dict,
                 PERIODO:str,
                 PERIODO_TRAIN1:str,
                 PERIODO_TRAIN2:str,
                 PERIODO_TRAIN3:str,
                 PERIODO_TRAIN4:str,
                 PERIODO_TRAIN5:str,
                 PERIODO_TRAIN6:str,
                 PERIODO_TEST1:str,
                 PERIODO_TEST2:str,
                 PERIODO_TEST3:str,
                 MODELO_PRODUCTIVO:str,
                 MODELO_PRODUCTIVO_param_test:dict,
                 GRABAR_BINARIOS:bool,
                 spark,
                 PATH:str='',
                 modelo_text:str='',
                 ambiente:str="sdb_datamining") -> None:
        
        self.BALANCEAR_TARGET=BALANCEAR_TARGET
        self.ELIMINAR_CORRELACIONES=ELIMINAR_CORRELACIONES
        self.CASTEAR_BIGINT=CASTEAR_BIGINT
        self.REDONDEAR_DECIMALES=REDONDEAR_DECIMALES
        self.CON_SCALER=CON_SCALER
        self.TIENE_TESTING=TIENE_TESTING
        self.CORRER_RF=CORRER_RF
        self.CORRER_GB=CORRER_GB
        self.CORRER_LGBM=CORRER_LGBM
        self.CORRER_XGB=CORRER_XGB
        self.CORRER_PRODUCTIVO=CORRER_PRODUCTIVO
        self.CAMPO_CLAVE=CAMPO_CLAVE
        self.TARGET=TARGET
        self.modelo=modelo
        self.ABT_VARIABLES=ABT_VARIABLES
        self.ABT_TABLA=ABT_TABLA
        self.TGT_TABLA=TGT_TABLA
        self.TGT_VARIABLES=TGT_VARIABLES
        self.TGT_BALENCEO=TGT_BALENCEO
        self.DECIMALES_VARIABLES_NUMERICAS=DECIMALES_VARIABLES_NUMERICAS
        self.COTA_CORRELACIONES=COTA_CORRELACIONES
        self.REGISTROS_X_PARTICION=REGISTROS_X_PARTICION
        self.PORCENTAJE_TRAINING=PORCENTAJE_TRAINING
        self.RF_param_test=RF_param_test
        self.MODELO_PRODUCTIVO=MODELO_PRODUCTIVO
        self.MODELO_PRODUCTIVO_param_test=MODELO_PRODUCTIVO_param_test
        self.GB_param_test=GB_param_test
        self.LGBM_param_test=LGBM_param_test
        self.XGB_param_test=XGB_param_test
   

        self.PERIODO=PERIODO
        self.PERIODO_TRAIN1=PERIODO_TRAIN1
        self.PERIODO_TRAIN2=PERIODO_TRAIN2
        self.PERIODO_TRAIN3=PERIODO_TRAIN3
        self.PERIODO_TRAIN4=PERIODO_TRAIN4
        self.PERIODO_TRAIN5=PERIODO_TRAIN5
        self.PERIODO_TRAIN6=PERIODO_TRAIN6
        self.PERIODO_TEST1=PERIODO_TEST1
        self.PERIODO_TEST2=PERIODO_TEST2
        self.PERIODO_TEST3=PERIODO_TEST3
        self.GRABAR_BINARIOS=GRABAR_BINARIOS
        self.spark=spark

        self.modelo_text=modelo_text
        self.ambiente=ambiente
        self.formato='parquet'
        ## Paths
        # Path HDFS
        self.PATH_MODELO = os.path.join("/adv/modelos/", modelo)
        self.PATH=os.path.join(self.PATH_MODELO, "challenger")
        # Path Local
        self.TMP_PATH=os.path.join("/tmp/adv/DM/datastore/DS_sandbox", modelo)


    def BorrarTablasTemporales(self ):

        try:
            print("INFORMACION DE SEGUIMIENTO: DROP TABLA _0")
            self.spark.sql(' drop table ' + self.ambiente + '.tmp_challenger_' + self.modelo + '_0 PURGE' )
        except:
            pass

        try:
            print("INFORMACION DE SEGUIMIENTO: DROP TABLA _1")
            self.spark.sql(' drop table ' + self.ambiente + '.tmp_challenger_' + self.modelo + '_1 PURGE'  )
        except:
            pass

        try:
            print("INFORMACION DE SEGUIMIENTO: DROP TABLA _2")
            self.spark.sql(' drop table ' + self.ambiente + '.tmp_challenger_' + self.modelo + '_2 PURGE'  )
        except:
            pass


        try:
            print("INFORMACION DE SEGUIMIENTO: DROP TABLA _testing")
            self.spark.sql(' drop table ' + self.ambiente + '.tmp_challenger_' + self.modelo + '_testing PURGE' )
        except:
            pass
        
        
        

    def BorrarResultadosDelModelo(self):    
        # Estas 2 son las que hay que grabar....
        
        try:
            print("INFORMACION DE SEGUIMIENTO: DROP TABLA _metricas")
            self.spark.sql(' drop table ' + self.ambiente + '.tmp_challenger_' + self.modelo + '_metricas PURGE' )
        except:
            pass
        
        
        try:
            print("INFORMACION DE SEGUIMIENTO: DROP TABLA _feature_importance")
            self.spark.sql(' drop table ' + self.ambiente + '.tmp_challenger_' + self.modelo + '_feature_importance PURGE' )
        except:
            pass
        
        
        try:
            print("INFORMACION DE SEGUIMIENTO: DROP TABLA _feature_importance_rank")
            self.spark.sql(' drop table ' + self.ambiente + '.tmp_challenger_' + self.modelo + '_feature_importance_rank PURGE' )
        except:
            pass

    def BalancearABT(self, train_df,  pBalanceo):
        
        
        ### Undersampling
        # Realizamos undersampling para balancear las clases 0 y 1 del target del dataset de training, quedando una relacion 1 a 20

        sample0 = train_df.filter(F.col(self.TARGET) == 0).count()
        sample1 = train_df.filter(F.col(self.TARGET) == 1).count()

        if(sample0>=sample1):
            major_df = train_df.filter(F.col(self.TARGET) == 0)
            minor_df = train_df.filter(F.col(self.TARGET) == 1)
        else:
            major_df = train_df.filter(F.col(self.TARGET) == 1)
            minor_df = train_df.filter(F.col(self.TARGET) == 0)


        ratio = int(major_df.count()/minor_df.count())

        sampled_majority_df = major_df.sample(False, pBalanceo/ratio, seed=1234)
        train_undersampled_df = sampled_majority_df.unionAll(minor_df)

        # Aca empieza el codigo de Sebastian
        print('Data Frame 1: ', train_undersampled_df.count())

        return train_undersampled_df
    
    def CastBigInt(self, train_undersampled_df):
    
        
        
        
        a = pd.DataFrame(train_undersampled_df.dtypes)
        a.columns = ['columna', 'tipo']
        
        print(a.tipo.value_counts())
        
        print(list(a[(a.tipo == 'bigint') & (a.columna != self.CAMPO_CLAVE)].columna))
        
        
        
        
        
        # Si no se quiere que todas las bigint pasen a ShortType cambiar variables_bigint
        
        variables_bigint = list(a[a.tipo == 'bigint'].columna)
        
        """
            Cuanto mas chico el nro mejor 
            https://spark.apache.org/docs/latest/sql-ref-datatypes.html
            Numeric types
                ByteType: Represents 1-byte signed integer numbers. The range of numbers is from -128 to 127.
                ShortType: Represents 2-byte signed integer numbers. The range of numbers is from -32768 to 32767.
                IntegerType: Represents 4-byte signed integer numbers. The range of numbers is from -2147483648 to 2147483647.
        """
            
        for c_name in variables_bigint :
            # print(c_name)
            train_undersampled_df = train_undersampled_df.withColumn(c_name, F.col(c_name).cast(ShortType()))
            
        return train_undersampled_df
    
    def RedondearDecimales(self, train_undersampled_df, pDecimales):
        # Redondeo decimales
        # Numerical vars
        numericCols = [c for c in train_undersampled_df.columns if c not in [self.CAMPO_CLAVE,'periodo', 'origin', 'label']]
        print("Num. numeric vars: " , len(numericCols))
        
        for c_name, c_type in train_undersampled_df.dtypes:
            #if c_type in ('double', 'float', 'decimal', 'int', 'smallint'):
            train_undersampled_df = train_undersampled_df.withColumn(c_name, F.round(c_name, pDecimales))
        
        return train_undersampled_df



    def EliminarCorrelaciones(self, train_undersampled_df, pCota):
            
        # Saco Columnas Correlacionadas

        # Numerical vars
        numericCols = [c for c in train_undersampled_df.columns if c not in [self.CAMPO_CLAVE,'periodo', 'origin', 'label']]
        
        print("Num. numeric vars: " , len(numericCols))
        
        # Saco correlaciones con un 10% de la base en Pandas, :(
        

        
        df = train_undersampled_df.sample(fraction=0.2, seed=1234).toPandas()
        
        # Create correlation matrix
        corr_matrix = df.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > pCota)]
        
        # Drop features 
        print(to_drop)
        # df.drop(to_drop, axis=1, inplace=True)
        
        
        print('*'*20)
        print('Variables a eliminar: ', len(to_drop))
        
        train_undersampled_df = train_undersampled_df.drop(*to_drop)
        
        print('Variables finales: ', len(train_undersampled_df.columns))
        
        return train_undersampled_df



    def ControlParticiones(self, train_undersampled_df:DataFrame, pCampoClave, Num_reg_particion):
        # Control de Particiones.....
      
               
        print(train_undersampled_df.count())
            
        # Particiono en 4 Partes, aca particionar en Pares dependiendo del tamaño, no muy bajo y no muy grande cada particion....
        # Mas de 100k y menos de 300k cada particion
        
        # Ni muy chica, ni muy grande cada particion
        
        Cantidad_de_Particiones_0 = int(train_undersampled_df.count() / Num_reg_particion)



        if ((Cantidad_de_Particiones_0 % 4) >=2):
            Cantidad_de_Particiones = int(Cantidad_de_Particiones_0/4)*4+4
        elif(Cantidad_de_Particiones_0 <= 1):
            Cantidad_de_Particiones = 1
        else:
            Cantidad_de_Particiones = int(Cantidad_de_Particiones_0/4)*4



        print(Cantidad_de_Particiones)



        train_undersampled_df = train_undersampled_df.repartition(Cantidad_de_Particiones, pCampoClave)
        train_undersampled_df.groupBy(F.spark_partition_id()).count().show()
        
        return train_undersampled_df    
    



    def EntrenarModeloSpark(self, pALGORITHM, train_undersampled_df, Training_Porcentaje, parametros, Nombre_Modelo):
    #Training_Porcentaje = 0.2
    #pALGORITHM = 'RF'
    #if 1 > 0:

        
        numTrees = parametros['numTrees']
        maxIter = parametros['maxIter']
        maxDepth = parametros['maxDepth']
        minInstancesPerNode  = parametros['minInstancesPerNode']
        maxBins = parametros['maxBins']
        
        
            
        if pALGORITHM == 'RF':
            print('RANDOM FOREST')
            
            if self.CON_SCALER == True:
                model = RandomForestClassifier(labelCol="label_", featuresCol="features_scaled", seed=12345)
            else:
                model = RandomForestClassifier(labelCol="label_", featuresCol="features", seed=12345)
            
            paramGrid = ParamGridBuilder() \
                .addGrid(model.numTrees, numTrees) \
                .addGrid(model.maxDepth, maxDepth) \
                .addGrid(model.minInstancesPerNode, minInstancesPerNode) \
                .addGrid(model.maxBins, maxBins) \
                .build()
        
        elif pALGORITHM == 'GB':
            print('Gradient BOOSTING')
            
            
            if self.CON_SCALER == True:
                model = GBTClassifier(labelCol="label_", featuresCol="features_scaled", seed=12345)
            else:
                model = GBTClassifier(labelCol="label_", featuresCol="features", seed=12345)
                
            paramGrid = ParamGridBuilder() \
                .addGrid(model.maxIter, maxIter) \
                .addGrid(model.maxDepth, maxDepth) \
                .addGrid(model.minInstancesPerNode, minInstancesPerNode) \
                .addGrid(model.maxBins, maxBins) \
                .build()



        
        #separo Train y Test
        # No entrenar con mas de 500k casos... 
        
        (trainingData, testData) = train_undersampled_df.randomSplit([Training_Porcentaje, (1 - Training_Porcentaje)], seed=1234)
        
        print("TRAIN Shape: " , trainingData.count(), ' - ', len(trainingData.columns))
        
            
        # Numerical vars
        numericCols = [c for c in trainingData.columns if c not in [self.CAMPO_CLAVE,'periodo', 'origin', 'label']]
        
        print("Num. numeric vars: " , len(numericCols))
        
        

        # Target
        target_st = StringIndexer(inputCol=self.TARGET, outputCol='label_')
        
        # Variables
        assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
        
        scaler = StandardScaler(inputCol='features', outputCol='features_scaled', withStd=True, withMean=False)
        
        evaluator=BinaryClassificationEvaluator()
        
        crossval2 = CrossValidator(estimator=model,
                                estimatorParamMaps=paramGrid,
                                evaluator=evaluator,
                                numFolds=3)  # use 3+ folds in practice
        
        if self.CON_SCALER == True:
            print('Con Scaler !!!!!!!!!!!!!!!!!!!!!')
            stages = [target_st, assembler, scaler, crossval2 ]
        else:
            
            print('Sin Scaler !!!!!!!!!!!!!!!!!!!!!')
            stages = [target_st, assembler, crossval2 ]
        
        pipeline = Pipeline(stages=stages)
        
        # Run cross-validation, and choose the best set of parameters.
        cvModel2 = pipeline.fit(trainingData)
        
        testDataScore = cvModel2.transform(testData)
        auc_cv = evaluator.evaluate(testDataScore, {evaluator.metricName: "areaUnderROC"})
        print('*'*20)
        print('auc 1 ', auc_cv)
        
        # El mejor modelo entrenado
        bestModel = cvModel2.stages[-1].bestModel
        
        hyperparametros = bestModel.extractParamMap()
        
        print("""
        ##############################################
        # Enteno el mejor Modelo
        ##############################################
        """)

        if pALGORITHM == 'RF':
                
            numTrees = bestModel.getOrDefault('numTrees')
            maxDepth = bestModel.getOrDefault('maxDepth')
            minInstancesPerNode = bestModel.getOrDefault('minInstancesPerNode')
            maxBins = bestModel.getOrDefault('maxBins')
            
            paramGrid3 = ParamGridBuilder() \
                .addGrid(model.numTrees, [numTrees]) \
                .addGrid(model.maxDepth, [maxDepth]) \
                .addGrid(model.minInstancesPerNode, [minInstancesPerNode]) \
                .addGrid(model.maxBins, [maxBins]) \
                .build()
            
        elif pALGORITHM == 'GB':
            
            maxIter = bestModel.getOrDefault('maxIter')
            maxDepth = bestModel.getOrDefault('maxDepth')
            minInstancesPerNode = bestModel.getOrDefault('minInstancesPerNode')
            maxBins = bestModel.getOrDefault('maxBins')
            
            paramGrid3 = ParamGridBuilder() \
                .addGrid(model.maxIter, [maxIter]) \
                .addGrid(model.maxDepth, [maxDepth]) \
                .addGrid(model.minInstancesPerNode, [minInstancesPerNode]) \
                .addGrid(model.maxBins, [maxBins]) \
                .build()
            

        crossval3 = CrossValidator(estimator=model,
                                estimatorParamMaps=paramGrid3,
                                evaluator=evaluator,
                                numFolds=3)  # use 3+ folds in practice
        
        if self.CON_SCALER == True:
            stages = [target_st, assembler, scaler, crossval3 ]
        else:
            stages = [target_st, assembler, crossval3 ]
            
        pipeline3 = Pipeline(stages=stages)
        
        # Run cross-validation, and choose the best set of parameters.
        cvModel3 = pipeline3.fit(trainingData)

        print("""
        ##########################################
        # Variables Importantes
        """)
        feat_imp = pd.DataFrame((cvModel3.stages[-1].bestModel.featureImportances.toArray()), index=numericCols).reset_index()
        feat_imp.columns =['variable', 'importance']
        
        
        SMALL_ALGO = Nombre_Modelo.lower()
        BINARIO = f"{self.PERIODO}_challenger_{SMALL_ALGO}"
        
        
        model_cvresults = self.spark.createDataFrame(
            feat_imp, 
            [ "variable", "importance"]  
        )
        
        model_cvresults = model_cvresults.withColumn("bin", F.lit(BINARIO))\
                                        .withColumn("periodo", F.lit(self.PERIODO))\
                                        .withColumn("algorithm", F.lit(Nombre_Modelo)) 
        

        try:
            a = self.spark.sql("select count(1) from " + self.ambiente + '.tmp_challenger_' + self.modelo + '_feature_importance')
            model_cvresults.write.mode('append').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_feature_importance')
        except:
            model_cvresults.write.mode('overwrite').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_feature_importance')
        
        
        
        #########################################################


        firstelement=udf(lambda v:float(v[1]),FloatType())

        trainingDataScore = cvModel3.transform(trainingData)
        trainingDataScore = trainingDataScore.withColumn('Prob1', firstelement('probability'))

        testDataScore = cvModel3.transform(testData)
        testDataScore = testDataScore.withColumn('Prob1', firstelement('probability'))
        
        auc_cv = evaluator.evaluate(testDataScore, {evaluator.metricName: "areaUnderROC"})
        print('*'*20)
        print('auc 2 ', auc_cv)
        
        print('*'*20)
        print('*'*20)
        
        print(cvModel3.stages[-1].bestModel.extractParamMap())
        ################################################
        
        df_values_lst = []
        df_values_lst.append((self.PERIODO, Nombre_Modelo  , "hyperparametros", str(hyperparametros)))
        df_values_lst.append((self.PERIODO, Nombre_Modelo , "AUC_VALIDACION", str(auc_cv)))

        model_cvresults = self.spark.createDataFrame(
            df_values_lst, 
            ["periodo", "algorithm", "metric_desc", "metric_value"]  
        )
        
        
        SMALL_ALGO = Nombre_Modelo.lower()
        BINARIO = f"{self.PERIODO}_challenger_{SMALL_ALGO}"
        
        # Add bin column
        model_cvresults = model_cvresults.withColumn("bin", F.lit(BINARIO))

        # Order columns
        model_cvresults = model_cvresults.select("algorithm", "metric_desc", "metric_value", "periodo", "bin")

        # Cambiar esto con la tabla original
        
        try:
            a = self.spark.sql("select count(1) from " + self.ambiente + '.tmp_challenger_' + self.modelo + '_metricas')
            model_cvresults.write.mode('append').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_metricas')
        except:
            model_cvresults.write.mode('overwrite').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_metricas')
        
        print("""
        ###############################################
        # Sacar para grabar
        """)

        ### Save model
        # Seleccionamos el mejor modelo y lo guardamos para compararlo con el otros modelos para luego elegir el modelo ganador
        if self.GRABAR_BINARIOS:

            # Crear carpeta HDFS
            os.popen(f'hadoop fs -mkdir -p {self.PATH}')
            # Guarda binario
            assert BINARIO
            path_binario = os.path.join(self.PATH, BINARIO + ".bin")
            cvModel3.write().overwrite().save(path_binario)    
            # Dar permisos
            os.popen(f"hadoop dfs -chmod -R 770 {self.PATH_MODELO}")


            
        try:
            self.CalcularDeciles(trainingDataScore.select(self.CAMPO_CLAVE, 'label', 'Prob1').toPandas(), testDataScore.select(self.CAMPO_CLAVE, 'label', 'Prob1').toPandas())
        except:
            print('Error Calcular Deciles')
        return cvModel3

    def TestingModeloSpark(self, pALGORITHM, testing_df, pModel_train, Nombre_Modelo):

        print("""    
        ###################################################
        # Scoreo
        """)

            
        evaluator=BinaryClassificationEvaluator()
        
        testDataScore_Val = pModel_train.transform(testing_df) 
        auc_cv = evaluator.evaluate(testDataScore_Val, {evaluator.metricName: "areaUnderROC"})
        print('*'*20)
        print('auc 2 ', auc_cv)
        
        print("""
        ###################################################
        # Grabo Resultados
        """)

        df_values_lst = []
        df_values_lst.append((self.PERIODO, Nombre_Modelo , "AUC_TESTEO", str(auc_cv)))
        
        model_cvresults = self.spark.createDataFrame(df_values_lst, ["periodo", "algorithm", "metric_desc", "metric_value"]  )
        SMALL_ALGO = Nombre_Modelo.lower()
        BINARIO = f"{self.PERIODO}_challenger_{SMALL_ALGO}"
        model_cvresults = model_cvresults.withColumn("bin", F.lit(BINARIO))
        model_cvresults = model_cvresults.select("algorithm", "metric_desc", "metric_value", "periodo", "bin")
        
        model_cvresults.write.mode('append').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_metricas')
        
        
    def EntrenarModeloPandas(self, pALGORITHM, train_undersampled_df, Training_Porcentaje, param_test, Nombre_Modelo):
    #Training_Porcentaje = 0.1
    #pALGORITHM = 'XGB'

    #if 1 > 0:

        
        
        #separo Train y Test
        # No entrenar con mas de 500k casos... 
        
        (trainingData, testData) = train_undersampled_df.randomSplit([Training_Porcentaje, (1-Training_Porcentaje)], seed=1234)
        
        print("TRAIN Shape: " , trainingData.count(), ' - ', len(trainingData.columns))
        
        
        ####################################################
        
        
        df = trainingData.toPandas()
        df['TGT'] = df['label'].astype(np.int)
        try:
            df.drop('label', axis=1, inplace=True)
        except:
            pass
        
        print("""
        ####################################################
        # Train y test
        """)

        X_train, X_test = train_test_split(df.copy(), test_size=0.3, random_state=42, stratify=df['TGT']);  
        
        print("""
        ###################################################
        # Standarizar 
        ###################################################
        """)

        # Get column names first
        
        names_df = self.spark.sql("select * from " + self.ambiente + '.tmp_challenger_' + self.modelo + "_variables where variable not in ( 'label' , '" + self.CAMPO_CLAVE + "') order by variable  ")
        numericCols = list(names_df.toPandas()['variable'])

        
        
        #numericCols = [c for c in df.columns if c not in [self.CAMPO_CLAVE,'periodo', 'origin', 'TGT', 'label']]
        print('columnas ', len(numericCols))
        
        numerical_cols = numericCols
        #numerical_cols = idx[(idx.str == False ) & (idx.col != 'TGT') & (idx.col != self.CAMPO_CLAVE)]['col']
        names = numerical_cols
        
        X_train[numerical_cols] = X_train[numerical_cols].astype(np.float64)
        X_test[numerical_cols] = X_test[numerical_cols].astype(np.float64)
        
        # Create the Scaler object
        scaler = preprocessing.StandardScaler(copy=True)
        
        if self.CON_SCALER == True:
            
            print('Con Scaler !!!!!!!!!!!!!!!!!!!!!')
            scaler.fit(X_train[names])
            # Fit your data on the scaler object
            scaled_est = scaler.transform(X_train[names])
            scaled_est = pd.DataFrame(scaled_est, columns=names, index=X_train.index)
            
            X_train.drop(names, axis=1, inplace = True)
            X_train2 = pd.concat((X_train, scaled_est), axis=1, sort=False)
            
            X_train2.head(1).T
            # test
            scaled_est_test = scaler.transform(X_test[names])
            scaled_est_test = pd.DataFrame(scaled_est_test, columns=names, index=X_test.index)
            X_test.drop(names, axis=1, inplace = True)
            X_test2 = pd.concat((X_test, scaled_est_test), axis=1, sort=False)
            
            X_train2.shape
            X_test2.shape
            
            X_train = X_train2.copy()
            X_test = X_test2.copy()
            
        

        
        
        target_column = 'TGT'
            
        
        print(numerical_cols)
        
        cross_val = StratifiedKFold(n_splits=10) 
            
        if pALGORITHM == 'LGBM':
            print('LGBM')
            #########################################################
            # Modelo 1
            #########################################################
            
            
            
            fit_params={"early_stopping_rounds":1000, 
                        "eval_metric" : 'auc', 
                        "eval_set" : [(X_test[numerical_cols],X_test[target_column])],
                        'verbose': 100
            }
            
            
            
            #This parameter defines the number of HP points to be tested
            
            #n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
            
            clf = lgb.LGBMClassifier(random_state=314, silent=True, metric='None',  
                                    #nfold=10, 
                                    n_jobs=4)
            
            
            
            gs = RandomizedSearchCV( estimator=clf, param_distributions=param_test, 
                                        n_iter=100,
                                        scoring='roc_auc',
                                        cv=cross_val,
                                        refit=True,
                                        random_state=314,
                                        verbose=True)
            
            gs.fit(X_train[numerical_cols],X_train[target_column], **fit_params)
            
            # principales variables
            feat_imp = pd.Series(gs.best_estimator_.feature_importances_, index=X_train[numerical_cols].columns)
            
            print("""
            ############################
            # El mejor modelo
            ############################
            """)

            opt_parameters = gs.best_estimator_.get_params()
            
            print(opt_parameters)
            
            #Configure from the HP optimisation
            def learning_rate_010_decay_power_0995(current_iter):
                base_learning_rate = 0.1
                lr = base_learning_rate  * np.power(.995, current_iter)
                return lr if lr > 1e-3 else 1e-3
            #clf_final = lgb.LGBMClassifier(**gs.best_estimator_.get_params())
            
            #Configure locally from hardcoded values
            clf_final = lgb.LGBMClassifier(**clf.get_params())
            print(clf.get_params())
            
            
            #set optimal parameters
            clf_final.set_params(**opt_parameters)
            
            #Train the final model with learning rate decay
            clf_final_train = clf_final.fit(X_train[ numerical_cols ], X_train[target_column],
                                            **fit_params, 
                                            callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])
            clf_final_train.best_score_
            
        
        elif pALGORITHM == 'XGB':
            print('XGBoost')
            
            

                
            xgb_model = xgb.XGBClassifier(objective='binary:logistic',
                                        seed = 1234,
                                        base_score = 0.5,
                                        booster = 'gbtree',
                                        gpu_id = -1,
                                        importance_type = 'gain',
                                        reg_alpha = 0.11,
                                        scale_pos_weight = 1,
                                        tree_method = 'exact',
                                        min_child_weight=0.6,
                                        colsample_bytree = 0.8,
                                        subsample = 0.85)
            
            gs = RandomizedSearchCV( estimator=xgb_model, 
                                        param_distributions=param_test, 
                                        n_iter=100,
                                        scoring='roc_auc',
                                        cv=cross_val,
                                        refit=True,
                                        random_state=314,
                                        verbose=True)
            
            gs.fit(X_train[numerical_cols],X_train[target_column])
            
            print("""
            ############################
            # El mejor modelo
            ############################
            """)

            opt_parameters = gs.best_estimator_.get_params()
            
            print(opt_parameters)
            
            
            xgb_model = xgb.XGBClassifier(objective='binary:logistic',
                                        seed = 1234,
                                        base_score = 0.5,
                                        booster = 'gbtree',
                                        gpu_id = -1,
                                        importance_type = 'gain',
                                        reg_alpha = 0.11,
                                        scale_pos_weight = 1,
                                        tree_method = 'exact',
                                        min_child_weight=0.6,
                                        colsample_bytree = 0.8,
                                        subsample = 0.85,
                                        gamma = gs.best_estimator_.get_params()['gamma'],
                                        max_depth = gs.best_estimator_.get_params()['max_depth'],
                                        n_estimators = gs.best_estimator_.get_params()['n_estimators'],
                                        learning_rate = gs.best_estimator_.get_params()['learning_rate'],
                                        
                                        )
                                        
                                        
                                        
            #Train the final model with learning rate decay
            clf_final_train = xgb_model.fit(X_train[ numerical_cols ], X_train[target_column] )

        print("""
        ##########################################
        # Variables Importantes
        """)

        feat_imp = pd.DataFrame(clf_final_train.feature_importances_, index=X_train[numerical_cols].columns).reset_index()
        feat_imp.columns =['variable', 'importance']
        
        
        SMALL_ALGO = Nombre_Modelo.lower()
        BINARIO = f"{self.PERIODO}_challenger_{SMALL_ALGO}"
        
        
        model_cvresults = self.spark.createDataFrame(
            feat_imp, 
            [ "variable", "importance"]  
        )
        
        model_cvresults = model_cvresults.withColumn("bin", F.lit(BINARIO))\
                                        .withColumn("periodo", F.lit(self.PERIODO))\
                                        .withColumn("algorithm", F.lit(Nombre_Modelo)) 
        
        
        

        try:
            a = self.spark.sql("select count(1) from " + self.ambiente + '.tmp_challenger_' + self.modelo + '_feature_importance')
            model_cvresults.write.mode('append').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_feature_importance')
        except:
            model_cvresults.write.mode('overwrite').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_feature_importance')
        
        
        ###############################################

        # results..
        
        probabilities_train = clf_final_train.predict_proba(X_train[numerical_cols])
        a = X_train[[target_column, self.CAMPO_CLAVE]].reset_index()
        a.columns = ['idx1', 'label', self.CAMPO_CLAVE]
        b = pd.DataFrame(probabilities_train[:,1], columns=['Prob1']).reset_index()
        trainDataScore = pd.concat([a, b], axis=1)
        


        probabilities = clf_final_train.predict_proba(X_test[numerical_cols])
        a = X_test[[target_column, self.CAMPO_CLAVE]].reset_index()
        a.columns = ['idx1', 'label', self.CAMPO_CLAVE]
        b = pd.DataFrame(probabilities[:,1], columns=['Prob1']).reset_index()
        testDataScore = pd.concat([a, b], axis=1)
        
        
        y_pred = clf_final_train.predict(X_test[numerical_cols])
        
        print("""
        ##############################################
        # ROC
        """)

        
        a = pd.DataFrame(X_test[[target_column, self.CAMPO_CLAVE]], columns=['TGT', self.CAMPO_CLAVE])
        a = a.reset_index()
        b = pd.DataFrame(probabilities[:,1], columns=['Prob1'])
        
        result = pd.concat([a, b], axis=1)
        
        yPred = y_pred
        yScore = result['Prob1']
        yTest = result['TGT']
        areaBajoCurvaRoc = roc_auc_score(yTest, yScore)
        accuracy = accuracy_score(yTest, yPred)
        
        print('ROC: ', areaBajoCurvaRoc)
        
        auc_cv = areaBajoCurvaRoc
        hyperparametros = opt_parameters
        
        ################################################
        
        df_values_lst = []
        df_values_lst.append((self.PERIODO, Nombre_Modelo  , "hyperparametros", str(hyperparametros)))
        df_values_lst.append((self.PERIODO, Nombre_Modelo , "AUC_VALIDACION", str(auc_cv)))

        model_cvresults = self.spark.createDataFrame(
            df_values_lst, 
            ["periodo", "algorithm", "metric_desc", "metric_value"]  
        )
        
        
        SMALL_ALGO = Nombre_Modelo.lower()
        BINARIO = f"{self.PERIODO}_challenger_{SMALL_ALGO}"
        
        # Add bin column
        model_cvresults = model_cvresults.withColumn("bin", F.lit(BINARIO))

        # Order columns
        model_cvresults = model_cvresults.select("algorithm", "metric_desc", "metric_value", "periodo", "bin")

        # Cambiar esto con la tabla original
        
        try:
            a = self.spark.sql("select count(1) from " + self.ambiente + '.tmp_challenger_' + self.modelo + '_metricas')
            model_cvresults.write.mode('append').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_metricas')
        except:
            model_cvresults.write.mode('overwrite').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_metricas')
        
        
        print("""
        ###############################################
        # Sacar para grabar
        """)
    
        ### Save model
        # Seleccionamos el mejor modelo y lo guardamos para compararlo con el otros modelos para luego elegir el modelo ganador
        if self.GRABAR_BINARIOS:

            # Checks if folder exists, otherwise, it is created.
            exists = os.path.exists(self.TMP_PATH) 
            if not exists:
                os.makedirs(self.TMP_PATH)

            # Crear carpeta HDFS
            os.popen(f'hadoop fs -mkdir -p {self.PATH}')
            

            assert BINARIO
            scaler_filename = BINARIO + "_scaler.bin"
            model_filename = BINARIO + "_model.bin"

            ## Save scaler
            # Save tmp file in local
            tmp_path_scaler = os.path.join(self.TMP_PATH, scaler_filename)
            pickle.dump(scaler, open( tmp_path_scaler, 'wb'))
            print(f"Guardado el archivo temporal a {tmp_path_scaler}")
            # Copy tmp file to hdfs
            path_scaler = os.path.join(self.PATH, scaler_filename)
            hdfsput = Popen(["hdfs", "dfs", "-copyFromLocal", "-f", 
                            tmp_path_scaler, 
                            path_scaler], 
                            stdin=PIPE, bufsize=-1)
            hdfsput.communicate()
            print(f"Copiado el archivo a {path_scaler}")
            # Delete tmp file
            if os.path.exists(tmp_path_scaler):
                os.remove(tmp_path_scaler)
            

            ## Save model
            # Save tmp file in local
            tmp_path_model = os.path.join(self.TMP_PATH, model_filename)
            pickle.dump(clf_final_train, open(tmp_path_model, 'wb'))
            print(f"Guardado el archivo temporal en {tmp_path_model}")
            # Copy tmp file to hdfs
            path_model = os.path.join(self.PATH, model_filename)
            hdfsput = Popen(["hdfs", "dfs", "-copyFromLocal", "-f", 
                            tmp_path_model, 
                            path_model], 
                            stdin=PIPE, bufsize=-1)
            hdfsput.communicate()
            print(f"Copiado el archivo a {path_model}")    
            # Delete tmp file
            if os.path.exists(tmp_path_scaler):
                os.remove(tmp_path_scaler)


            # Dar permisos
            os.popen(f"hadoop dfs -chmod -R 770 {self.PATH_MODELO}")


        try:
            
            self.CalcularDeciles(trainDataScore[[self.CAMPO_CLAVE, 'label', 'Prob1']], testDataScore[[self.CAMPO_CLAVE, 'label', 'Prob1']])
        except:
            print('Error Calcular Deciles')
        return scaler, clf_final_train

    def TestingModeloPython(self, pALGORITHM, testing_df, pScaler_train, pModel_train, Nombre_Modelo ):
    
    #pALGORITHM = 'XGB'
    #pScaler_train = XGB_scaler_train
    #pModel_train = XGB_model_train

    # if 1 > 10:



        # Leo las variables de entrada al self.modelo
        
        names_df = self.spark.sql("select * from " + self.ambiente + '.tmp_challenger_' + self.modelo + "_variables where  variable not in ( 'label' , '" + self.CAMPO_CLAVE + "')  order by variable  ")
        names = list(names_df.toPandas()['variable'])

        
        
        
        ###################################################
        
        X_test = testing_df.select(*names).toPandas()
        
        
        print(X_test.shape)
        
        ###################################################
        
        print(len(names))
        
        X_test[names] = X_test[names].astype(np.float64)
        
        if self.CON_SCALER == True:
            scaled_est_test = pScaler_train.transform(X_test[names])
            scaled_est_test = pd.DataFrame(scaled_est_test, columns=names, index=X_test.index)
            X_test = scaled_est_test.copy()
        
        #Todo: Verfificar validez del siguiente if, los imports se movieron al principio del archivo
        if pALGORITHM == 'LGBM':
            print('LGBM')       

        elif pALGORITHM == 'XGB':
            print('XGBoost')
        
        y_test = testing_df.select('label').toPandas()

        probabilities       = pModel_train.predict_proba(X_test[names])
        y_pred              = pModel_train.predict(X_test[names])

        print("""
        ##############################################
        # ROC
        """)
        
        a = pd.DataFrame(y_test[['label']], columns=['label'])
        a = a.reset_index()
        b = pd.DataFrame(probabilities[:,1], columns=['Prob1'])
        
        result = pd.concat([a, b], axis=1)
        
        yPred = y_pred
        yScore = result['Prob1']
        yTest = result['label']
        auc_cv = roc_auc_score(yTest, yScore)
        accuracy = accuracy_score(yTest, yPred)
        
        print('ROC : ', auc_cv)
        
        ###################################################
        # Grabo Resultados
        
        df_values_lst = []
        df_values_lst.append((self.PERIODO, Nombre_Modelo  , "AUC_TESTEO", str(auc_cv)))
        
        model_cvresults = self.spark.createDataFrame(df_values_lst, ["periodo", "algorithm", "metric_desc", "metric_value"]  )

        SMALL_ALGO = Nombre_Modelo.lower()
        BINARIO = f"{self.PERIODO}_challenger_{SMALL_ALGO}"

        model_cvresults = model_cvresults.withColumn("bin", F.lit(BINARIO))
        model_cvresults = model_cvresults.select("algorithm", "metric_desc", "metric_value", "periodo", "bin")

        model_cvresults.write.mode('append').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_metricas')
        
    def CalcularDeciles(self, pTrain, pTest):
        

        
        ###############################################
        print('Training')
        result = pTrain
        
        result['porc'] = result['Prob1'].rank(pct=True) * 100
        
        
        result.loc[result['porc'].between(0, 10, inclusive=False), 'decil'] = 10
        result.loc[result['porc'].between(10, 20, inclusive=True), 'decil'] = 9
        result.loc[result['porc'].between(20, 30, inclusive=False), 'decil'] = 8
        result.loc[result['porc'].between(30, 40, inclusive=True), 'decil'] = 7
        result.loc[result['porc'].between(40, 50, inclusive=False), 'decil'] = 6
        result.loc[result['porc'].between(50, 60, inclusive=True), 'decil'] = 5
        result.loc[result['porc'].between(60, 70, inclusive=False), 'decil'] = 4
        result.loc[result['porc'].between(70, 80, inclusive=True), 'decil'] = 3
        result.loc[result['porc'].between(80, 90, inclusive=False), 'decil'] = 2
        result.loc[result['porc'].between(90, 101, inclusive=True), 'decil'] = 1
        
        print(result.decil.value_counts())
        print(result[result.label == 1].decil.value_counts())
        
        a = result.groupby('decil')['Prob1'].agg(min)
        print(a)
        
        
        deciles = pd.DataFrame(result.groupby('decil')['Prob1'].min().reset_index())
        deciles.columns = ['decil', 'cota']

        ##############################################
        
        result = pTest
        print('*'*20)
        print('Testing')
        result['decil'] = np.where(result.Prob1 >= deciles[deciles.decil == 1]['cota'][0]                                  , 1,  
                                np.where((result.Prob1 >=  deciles[deciles.decil == 2]['cota'][1]) & (result.Prob1 < deciles[deciles.decil == 1]['cota'][0] ), 2,
                                np.where((result.Prob1 >=  deciles[deciles.decil == 3]['cota'][2]) & (result.Prob1 < deciles[deciles.decil == 2]['cota'][1] ) , 3,
                                np.where((result.Prob1 >=  deciles[deciles.decil == 4]['cota'][3] ) & (result.Prob1 < deciles[deciles.decil == 3]['cota'][2]), 4,
                                np.where((result.Prob1 >=  deciles[deciles.decil == 5]['cota'][4] ) & (result.Prob1 < deciles[deciles.decil == 4]['cota'][3]), 5,
                                np.where((result.Prob1 >=  deciles[deciles.decil == 6]['cota'][5] ) & (result.Prob1 < deciles[deciles.decil == 5]['cota'][4]), 6,
                                np.where((result.Prob1 >=  deciles[deciles.decil == 7]['cota'][6] ) & (result.Prob1 < deciles[deciles.decil == 6]['cota'][5]) , 7,
                                np.where((result.Prob1 >=  deciles[deciles.decil == 8]['cota'][7] ) & (result.Prob1 < deciles[deciles.decil == 7]['cota'][6]), 8,
                                np.where((result.Prob1 >=  deciles[deciles.decil == 9]['cota'][8] ) & (result.Prob1 < deciles[deciles.decil == 8]['cota'][7]), 9,
                                10)))))))))

        print(result.decil.value_counts())
        print(result[result.label == 1].decil.value_counts())





    def CrearTablas(self):
        ct = datetime.datetime.now()
        print("Crear Tablas:-", ct)
        self.BorrarTablasTemporales()

        print("""INFORMACION DE SEGUIMIENTO:
        ####################################################################
        # 1. Crear tabla para Training
        # 1.1. Leer ABT + TGT
        ####################################################################
        """)
              
        pABT    =   " SELECT a." + self.CAMPO_CLAVE + ", " + self.ABT_VARIABLES + " , coalesce(" + self.TGT_VARIABLES + """, 0) as label   
                    FROM """ + self.ABT_TABLA +  """ a """ +  """ 
                            left join """ + self.TGT_TABLA + " b  on a." + self.CAMPO_CLAVE + " = b." + self.CAMPO_CLAVE + """
                                                            AND a.periodo = b.periodo 
                    WHERE   a.periodo IN (""" + str(self.PERIODO_TRAIN1) + " , " +  str(self.PERIODO_TRAIN2)  + " , " +str( self.PERIODO_TRAIN3)  + " , " + str(self.PERIODO_TRAIN4)  + " , " + str(self.PERIODO_TRAIN5)  + " , " + str(self.PERIODO_TRAIN6) + ")"
                    
        print(pABT)
        
        # cargamos las variables de la abt que se uso en el modelo productivo y el target con los periodos de training para realizar el entrenamiento 
        train_undersampled_df = self.spark.sql(pABT).na.fill(-999)
        
        train_undersampled_df.write.mode('overwrite').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_0')
        
        print("""INFORMACION DE SEGUIMIENTO:
        ####################################################################
        # 1.2. Balanceo y Particiones
        ####################################################################
        """)

        train_undersampled_df = self.spark.sql("select * from " + self.ambiente + '.tmp_challenger_' +  self.modelo + '_0')
        print('TABLA ORIGINAL: ', train_undersampled_df.count())
        
        if self.BALANCEAR_TARGET  == True:
            train_undersampled_df = self.BalancearABT(train_undersampled_df, self.TGT_BALENCEO)
        
        train_undersampled_df = self.ControlParticiones(train_undersampled_df, self.CAMPO_CLAVE, self.REGISTROS_X_PARTICION) 
        
        
        train_undersampled_df.write.mode('overwrite').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_1')
        
        print("""INFORMACION DE SEGUIMIENTO:
        ####################################################################
        # 1.3. Corregir Numeros, Eliminar Correlaciones y Particiones
        ####################################################################
        """)

        train_undersampled_df = self.spark.sql("select * from " + self.ambiente + '.tmp_challenger_' +  self.modelo + '_1')
        
        train_undersampled_df = self.DecimalToDouble(train_undersampled_df)
        if self.CASTEAR_BIGINT == True: 
            train_undersampled_df = self.CastBigInt(train_undersampled_df)
            
        if self.REDONDEAR_DECIMALES == True:
            train_undersampled_df = self.RedondearDecimales(train_undersampled_df, self.DECIMALES_VARIABLES_NUMERICAS)
            
        if self.ELIMINAR_CORRELACIONES == True:
            train_undersampled_df = self.EliminarCorrelaciones(train_undersampled_df, self.COTA_CORRELACIONES)
        
        
        train_undersampled_df = self.ControlParticiones(train_undersampled_df, self.CAMPO_CLAVE, self.REGISTROS_X_PARTICION) 
        train_undersampled_df.write.mode('overwrite').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_2' )

        print("""INFORMACION DE SEGUIMIENTO:
        ####################################################################
        # 1.4. Grabo las variables que van a entrar al modelo
        ####################################################################
        """)

        # Grabar las variables.....
        columns = ['variable']
        
        variable = pd.DataFrame(train_undersampled_df.columns)
        variable.columns = ['variable']
        
        variable = self.spark.createDataFrame(
                variable, 
                ["variable"]  
            )
            
        variable.write.mode('overwrite').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_variables' )

        print("""INFORMACION DE SEGUIMIENTO:
        ####################################################################
        ## 2. Cargar ABT de Testing
        ####################################################################
        """)

        if self.TIENE_TESTING == True:
            
            pABT    =   " SELECT a.periodo , a." + self.CAMPO_CLAVE + ", " + self.ABT_VARIABLES + " , coalesce(" + self.TGT_VARIABLES + """, 0) as label   
                        FROM """ + self.ABT_TABLA +  """ a """ +  """ 
                                left join """ + self.TGT_TABLA + " b  on a." + self.CAMPO_CLAVE + " = b." + self.CAMPO_CLAVE + """
                                                                AND a.periodo = b.periodo 
                        WHERE   a.periodo IN (""" + str(self.PERIODO_TEST1) + " , " +  str(self.PERIODO_TEST2)  + " , " +str( self.PERIODO_TEST3) + ")"
            
            test_undersampled_df = self.spark.sql(pABT)
            test_undersampled_df = self.spark.sql(pABT).na.fill(-999)
            
            print('TABLA PREDICCIONES: ', test_undersampled_df.count())
            
            #############################################################################
            # estos pasos tienen que ser los mismos que los realizados en la ABT de Training
            
            train_undersampled_df = self.DecimalToDouble(train_undersampled_df)
            if self.CASTEAR_BIGINT == True:
                test_undersampled_df = self.CastBigInt(test_undersampled_df)
                
            if self.REDONDEAR_DECIMALES == True:
                test_undersampled_df = self.RedondearDecimales(test_undersampled_df, self.DECIMALES_VARIABLES_NUMERICAS)
            
            
            train_undersampled_df = self.ControlParticiones(train_undersampled_df, self.CAMPO_CLAVE, self.REGISTROS_X_PARTICION)
            test_undersampled_df.write.mode('overwrite').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_testing' )
        
        
    
    def LeerTablas(self):

        print("""INFORMACION DE SEGUIMIENTO:
        #############################################################################    
        # 3. Entreno el Modelo        
        #############################################################################
        """)

        train_undersampled_df = self.spark.sql("select * from " + self.ambiente + '.tmp_challenger_' + self.modelo + '_2')
        
        if self.TIENE_TESTING == True:
            testing_df_m1 = self.spark.sql("select * from " + self.ambiente + '.tmp_challenger_' + self.modelo + '_testing where periodo = ' + str(self.PERIODO_TEST1) )
            testing_df_m2 = self.spark.sql("select * from " + self.ambiente + '.tmp_challenger_' + self.modelo + '_testing where periodo = ' + str(self.PERIODO_TEST2) )
            testing_df_m3 = self.spark.sql("select * from " + self.ambiente + '.tmp_challenger_' + self.modelo + '_testing where periodo = ' + str(self.PERIODO_TEST3) )

        return train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3



    def RandomForest(self, Corrida, train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3):        
        
        print ('*'*30)
        print (' RANDOM FOREST ')
        Nombre_Modelo = 'RF_' + str(Corrida)
        RF_Model_train = self.EntrenarModeloSpark('RF', train_undersampled_df, self.PORCENTAJE_TRAINING, self.RF_param_test, Nombre_Modelo  ) 
        
        if self.TIENE_TESTING == True:        
            print( 'Testing ')
            self.TestingModeloSpark('RF', testing_df_m1, RF_Model_train, Nombre_Modelo + ' TESTING MES1')
            self.TestingModeloSpark('RF', testing_df_m2, RF_Model_train, Nombre_Modelo + ' TESTING MES2')
            self.TestingModeloSpark('RF', testing_df_m3, RF_Model_train, Nombre_Modelo + ' TESTING MES3')
         


    def GradientBoosting(self, Corrida, train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3):

        print ('*'*30)
        print (' GRADIENT BOOSTING ')

        Nombre_Modelo = 'GB_' + str(Corrida)             
        GB_Model_train = self.EntrenarModeloSpark('GB', train_undersampled_df, self.PORCENTAJE_TRAINING, self.GB_param_test, Nombre_Modelo) 
        
        if self.TIENE_TESTING == True:        
            print( 'Testing ')
            self.TestingModeloSpark('GB', testing_df_m1, GB_Model_train, Nombre_Modelo + ' TESTING MES1')
            self.TestingModeloSpark('GB', testing_df_m2, GB_Model_train, Nombre_Modelo + ' TESTING MES2')
            self.TestingModeloSpark('GB', testing_df_m3, GB_Model_train, Nombre_Modelo + ' TESTING MES3')
         


    def LIGHTGBMPandas(self, Corrida, train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3):
                        
        print ('*'*30)
        print (' LIGHTGBM ')
        
        Nombre_Modelo = 'LGBM_' + str(Corrida)
        LGBM_scaler_train, LGBM_model_train = self.EntrenarModeloPandas('LGBM', train_undersampled_df, self.PORCENTAJE_TRAINING, self.LGBM_param_test, Nombre_Modelo)
        
        if self.TIENE_TESTING == True:        
            print( 'Testing ')
            self.TestingModeloPython('LGBM', testing_df_m1, LGBM_scaler_train, LGBM_model_train, Nombre_Modelo + ' TESTING MES1')
            self.TestingModeloPython('LGBM', testing_df_m2, LGBM_scaler_train, LGBM_model_train, Nombre_Modelo + ' TESTING MES2')
            self.TestingModeloPython('LGBM', testing_df_m3, LGBM_scaler_train, LGBM_model_train, Nombre_Modelo + ' TESTING MES3')
          

    def XGBoostPandas(self, Corrida, train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3):
        
        print ('*'*30)
        print (' XGB ')
        
        Nombre_Modelo = 'XGB_' + str(Corrida)
        XGB_scaler_train, XGB_model_train = self.EntrenarModeloPandas('XGB', train_undersampled_df, self.PORCENTAJE_TRAINING, self.XGB_param_test, Nombre_Modelo)
        
        if self.TIENE_TESTING == True:        
            print( 'Testing ')
            self.TestingModeloPython('XGB', testing_df_m1, XGB_scaler_train, XGB_model_train, Nombre_Modelo + ' TESTING MES1')
            self.TestingModeloPython('XGB', testing_df_m2, XGB_scaler_train, XGB_model_train, Nombre_Modelo + ' TESTING MES2')
            self.TestingModeloPython('XGB', testing_df_m3, XGB_scaler_train, XGB_model_train, Nombre_Modelo + ' TESTING MES3')
                    


    def ModeloProductivo(self, Corrida, train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3):
        
        print ('*'*30)
        print (' MODELO PRODUCTIVO ')
        
        if self.MODELO_PRODUCTIVO == 'RF':
        
            Nombre_Modelo = 'RF_PRODUCTIVO' + str(Corrida)           
            Productivo_Model_train = self.EntrenarModeloSpark('RF', train_undersampled_df, self.PORCENTAJE_TRAINING, self.MODELO_PRODUCTIVO_param_test, Nombre_Modelo) 
        
        elif self.MODELO_PRODUCTIVO == 'GB':
            
            Nombre_Modelo = 'GB_PRODUCTIVO'   + str(Corrida)                      
            Productivo_Model_train = self.EntrenarModeloSpark('GB', train_undersampled_df, self.PORCENTAJE_TRAINING, self.MODELO_PRODUCTIVO_param_test, Nombre_Modelo) 
        
        elif self.MODELO_PRODUCTIVO == 'LGBM':
                
            Nombre_Modelo = 'LGBM_PRODUCTIVO' + str(Corrida)           
            Productivo_scaler_train, Productivo__model_train = self.EntrenarModeloPandas('LGBM', train_undersampled_df, self.PORCENTAJE_TRAINING, self.MODELO_PRODUCTIVO_param_test, Nombre_Modelo)
            
        elif self.MODELO_PRODUCTIVO == 'XGB':
                
            Nombre_Modelo = 'XGB_PRODUCTIVO' + str(Corrida)           
            Productivo_scaler_train, Productivo__model_train, XGB_trainingData_Score, XGB_testingData_Score = self.EntrenarModeloPandas('XGB', train_undersampled_df, self.PORCENTAJE_TRAINING, self.MODELO_PRODUCTIVO_param_test, Nombre_Modelo)
            
            self.CalcularDeciles(XGB_trainingData_Score, XGB_testingData_Score)
            
        if self.TIENE_TESTING == True:        
            print( 'Testing ')
            if self.MODELO_PRODUCTIVO == 'RF' or self.MODELO_PRODUCTIVO == 'GB':
                
                self.TestingModeloSpark(self.MODELO_PRODUCTIVO, testing_df_m1, Productivo_Model_train, Nombre_Modelo + ' TESTING MES1')
                self.TestingModeloSpark(self.MODELO_PRODUCTIVO, testing_df_m2, Productivo_Model_train, Nombre_Modelo + ' TESTING MES2')
                self.TestingModeloSpark(self.MODELO_PRODUCTIVO, testing_df_m3, Productivo_Model_train, Nombre_Modelo + ' TESTING MES3')
                    
            elif self.MODELO_PRODUCTIVO == 'XGB' or self.MODELO_PRODUCTIVO == 'LGBM':
                self.TestingModeloPython(self.MODELO_PRODUCTIVO, testing_df_m1, Productivo_scaler_train, Productivo__model_train, Nombre_Modelo + ' TESTING MES1')
                self.TestingModeloPython(self.MODELO_PRODUCTIVO, testing_df_m2, Productivo_scaler_train, Productivo__model_train, Nombre_Modelo + ' TESTING MES2')
                self.TestingModeloPython(self.MODELO_PRODUCTIVO, testing_df_m3, Productivo_scaler_train, Productivo__model_train, Nombre_Modelo + ' TESTING MES3')
        

    def MejoresVariables(self):
        print("""
        #############################################################################    
        # 4. Mejores Variables de Todos los modelos
        #############################################################################
        """)

        a = self.spark.sql("""select variable, sum(rownum ) as rownum 
                        from (select ROW_NUMBER() OVER(PARTITION BY algorithm ORDER BY importance DESC) AS rownum, * 
                                from """ + self.ambiente + '.tmp_challenger_' + self.modelo + """_feature_importance) 
                        group by variable
                        order by 2 """)
                    
        a.write.mode('overwrite').format(self.formato).saveAsTable(self.ambiente + '.tmp_challenger_' + self.modelo + '_feature_importance_rank')



    def EntrenarModelos(self, Corrida):
        ct = datetime.datetime.now()

        print("Entrenamiento:-", ct)
    
        train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3 = self.LeerTablas()


        # 3.1. Random Forest
        try:
            if self.CORRER_RF == True:
                self.RandomForest(Corrida, train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3)
        except:
            print('Error Random Forest') 


        # 3.2. Gradient Boosting 
        try:
            if self.CORRER_GB  == True:
                self.GradientBoosting(Corrida, train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3)
        except:
            print('Error Gradient Boosting')         
        
        
        ## 3.3. LIGHTGBM Pandas
        try:    
            if self.CORRER_LGBM  == True:
                self.LIGHTGBMPandas(Corrida, train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3)
        except:
            print('Error Lightgbm') 
    

        # 3.4. XGBoost Pandas
        try:
            if self.CORRER_XGB == True:
                self.XGBoostPandas(Corrida, train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3)
        except:
            print('Error XGBoost')
            

        # 3.5. Modelo Productivo
        try:            
            if self.CORRER_PRODUCTIVO == True:
                self.ModeloProductivo(Corrida, train_undersampled_df, testing_df_m1, testing_df_m2, testing_df_m3)
        except:
            print('Error Modelo Productivo ')
        

        ct = datetime.datetime.now()
        print("Entrenamiento Fin:-", ct)

        self.MejoresVariables()    
        
        

    def MejorModeloEntrenado(self, ):
        print("""
        #############################################################################    
        # 5. Elijo el Mejor modelo entrenado
        #############################################################################
        """)

        # Busco quien es el mejor en Testing
        
        meses_testing = self.spark.sql("""
            select  replace(replace(replace(A.ALGORITHM, 'MES1', ''), 'MES2', ''), 'MES3', '') AS ALGORITHM,
                    SUM(  case when mes1.ALGORITHM = a.ALGORITHM then 1 else 0 end
                        + case when mes2.ALGORITHM = a.ALGORITHM then 1 else 0 end
                        + case when mes3.ALGORITHM = a.ALGORITHM then 1 else 0 end ) as meses_ganadores
                        
                        
            FROM """ + self.ambiente + '.tmp_challenger_' + self.modelo +  """_metricas  A
                left join (SELECT MAX(ALGORITHM)  AS ALGORITHM /* PONGO UN MAX POR SI EMPATAN QUE SE QUEDE CON UNO */
                            FROM """ + self.ambiente + '.tmp_challenger_' + self.modelo +  """_metricas  A,
                                    (select  max(metric_value) AS metric_value
                                    from """ + self.ambiente + '.tmp_challenger_' + self.modelo +  """_metricas 
                                    where algorithm like '% TESTING MES1'
                                    AND   metric_desc = 'AUC_TESTEO' ) B
                            WHERE   A.metric_value = B.metric_value
                                    ) mes1  on a.ALGORITHM = mes1.ALGORITHM 
                
                left join (SELECT MAX(ALGORITHM)  AS ALGORITHM /* PONGO UN MAX POR SI EMPATAN QUE SE QUEDE CON UNO */
                            FROM """ + self.ambiente + '.tmp_challenger_' + self.modelo +  """_metricas  A,
                                    (select  max(metric_value) AS metric_value
                                    from """ + self.ambiente + '.tmp_challenger_' + self.modelo +  """_metricas 
                                    where algorithm like '% TESTING MES2'
                                    AND   metric_desc = 'AUC_TESTEO' ) B
                            WHERE   A.metric_value = B.metric_value
                                    ) mes2  on a.ALGORITHM = mes2.ALGORITHM 
                            
                left join (SELECT MAX(ALGORITHM)  AS ALGORITHM /* PONGO UN MAX POR SI EMPATAN QUE SE QUEDE CON UNO */
                            FROM """ + self.ambiente + '.tmp_challenger_' + self.modelo +  """_metricas  A,
                                    (select  max(metric_value) AS metric_value
                                    from """ + self.ambiente + '.tmp_challenger_' + self.modelo +  """_metricas 
                                    where algorithm like '% TESTING MES3'
                                    AND   metric_desc = 'AUC_TESTEO' ) B
                            WHERE   A.metric_value = B.metric_value
                                    ) mes3  on a.ALGORITHM = mes3.ALGORITHM 
                WHERE A.ALGORITHM LIKE '%TESTING MES%'
                GROUP BY replace(replace(replace(A.ALGORITHM, 'MES1', ''), 'MES2', ''), 'MES3', '')                                 
            """)
        
        print('*'*20)
        print('mejor modelo en Testing')
        meses_testing.show()
        
        meses_testing.createOrReplaceTempView("meses_testing")
        
        
        modelo_ganador_testing = self.spark.sql("""
            select  a.*
            FROM """ + self.ambiente + '.tmp_challenger_' + self.modelo +  """_metricas  A,
                        (SELECT MAX(ALGORITHM) AS ALGORITHM
                        FROM 
                            (select replace(replace(replace(A.ALGORITHM, 'MES1', ''), 'MES2', ''), 'MES3', '') AS ALGORITHM
                            from   meses_TESTING A,
                                    (SELECT MAX(MESES_GANADORES) AS MESES_GANADORES
                                    FROM MESES_TESTING
                                    ) B
                            WHERE  A.MESES_GANADORES = B.MESES_GANADORES
                            )
                        ) B
                WHERE A.ALGORITHM LIKE '%TESTING MES%'
                AND   replace(replace(replace(A.ALGORITHM, 'MES1', ''), 'MES2', ''), 'MES3', '') = B.ALGORITHM
                
            """)
        
        modelo_ganador_testing.show()
        
        modelo_ganador_testing.createOrReplaceTempView("modelo_ganador_testing")
        
        
        meses_ganadores_vs_produccion = self.spark.sql("""
            SELECT  SUM(MESES_GANADORES) AS MESES_GANADORES
            FROM (
                SELECT  A.*,
                            B.metric_value,
                            CASE WHEN B.metric_value - A.AUC_PROD > 0.02 THEN 1 ELSE 0 END AS MESES_GANADORES
                    from 
                            (SELECT  modelo,
                                    substr(cast(fecha AS STRING),1,6) AS periodo, 
                                    sum(suma_area) AS auc_prod    
                            FROM    data_lake_analytics.indicadores_performance  
                            WHERE   substr(cast(fecha AS STRING),1,6) IN ( """ + str(self.PERIODO_TEST1) + ',' + str(self.PERIODO_TEST2) + ',' + str(self.PERIODO_TEST3) + """)
                            AND     modelo = '""" + self.modelo + """'
                            AND     tipo = 'PERFORMANCE'
                            group by modelo, substr(cast(fecha AS STRING),1,6) ) A
                            
                            LEFT JOIN modelo_ganador_testing B ON A.PERIODO = case when b.algorithm like '% TESTING MES1' then """ + str(self.PERIODO_TEST1) + """
                                                                                    when b.algorithm like '% TESTING MES2' then """ + str(self.PERIODO_TEST2) + """
                                                                                    when b.algorithm like '% TESTING MES3' then """ + str(self.PERIODO_TEST3) + """
                                                                            end           
                ) sa
            """).toPandas()['MESES_GANADORES'][0]
            
        print('MESES QUE EL NUEVO MODELO LE GANA AL PRODUCTIVO: ', meses_ganadores_vs_produccion)


        prod_vs_chall = self.spark.sql("""
            SELECT  A.*,
                        B.metric_value as auc_testing, B.metric_value - A.AUC_PROD as diff,
                        CASE WHEN B.metric_value - A.AUC_PROD > 0.02 THEN 1 ELSE 0 END AS MESES_GANADORES
                from 
                        (SELECT  
                                
                                substr(cast(fecha AS STRING),1,6) AS periodo, 
                                sum(suma_area) AS auc_prod    
                        FROM    data_lake_analytics.indicadores_performance  
                        WHERE   substr(cast(fecha AS STRING),1,6) IN ( """ + str(self.PERIODO_TEST1) + ',' + str(self.PERIODO_TEST2) + ',' + str(self.PERIODO_TEST3) + """)
                        AND     modelo = '""" + self.modelo + """'
                        AND     tipo = 'PERFORMANCE'
                        group by modelo, substr(cast(fecha AS STRING),1,6) ) A
                        
                        LEFT JOIN modelo_ganador_testing B ON A.PERIODO = case when b.algorithm like '% TESTING MES1' then """ + str(self.PERIODO_TEST1) + """
                                                                                when b.algorithm like '% TESTING MES2' then """ + str(self.PERIODO_TEST2) + """
                                                                                when b.algorithm like '% TESTING MES3' then """ + str(self.PERIODO_TEST3) + """
                                                                        end
        """)

        prod_vs_chall.show(10)

        return meses_ganadores_vs_produccion, prod_vs_chall
        
        
    def DecimalToDouble(self, train_undersampled_df):
    
    
        numericCols = [c for c in train_undersampled_df.columns if c not in [self.CAMPO_CLAVE,'periodo', 'origin', 'label']]
        print("Num. numeric vars: " , len(numericCols))
        
        for c_name, c_type in train_undersampled_df.dtypes:
            if (c_type.find('decimal') >=0):
                train_undersampled_df = train_undersampled_df.withColumn(c_name, F.col(c_name).cast('double'))
        
        return train_undersampled_df    
            
            
    def EjecutarChallenger(self):
    #if 1 > 0:
        import datetime;
        ct = datetime.datetime.now()
        print("Challenger Inicio:-", ct)
        
        
        self.BorrarResultadosDelModelo()
        
        self.CrearTablas() 
        
        ct = datetime.datetime.now()

        self.EntrenarModelos(1)

        self.MejorModeloEntrenado()
        
        ct = datetime.datetime.now()
        print("Challenger Fin:-", ct)
        

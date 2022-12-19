import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ShortType
from pyspark.sql import DataFrame
import numpy as np
# from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
# from pyspark.ml.tuning import ParamGridBuilder
# from pyspark.ml.classification import GBTClassifier 
# from pyspark.sql.functions import udf
# from pyspark.sql.types import FloatType
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split
# from pyspark.ml.tuning import CrossValidator
# from pyspark.ml import Pipeline
# from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.classification import RandomForestClassifier
# import lightgbm as lgb
# import xgboost as xgb
# from sklearn.model_selection import GridSearchCV #Todo: Eliminar, No se usa
# from sklearn import preprocessing
# from sklearn.metrics import roc_auc_score, accuracy_score
import datetime;
class ABT():
 
    def __init__(self,
                 MODELO:str,
                 
                 periodo:str,
                 
                 TABLA_UNIVERSO:str,
                 
                 ELIMINAR_CORRELACIONES:bool, 
                 CASTEAR_BIGINT:bool, 
                 REDONDEAR_DECIMALES:bool, 
                 
                 CAMPO_CLAVE:str,
                 AGRUPAR:bool,
                 CAMPO_AGRUPAR:str,
                 
                 DECIMALES_VARIABLES_NUMERICAS:int, 
                 COTA_CORRELACIONES:float, 
                 REGISTROS_X_PARTICION:float, 
                 
                 COTA_REPRESENTATIVIDAD:int,
                 
                 pTablaPerfinesPospagoNivles:str,
                 pTablaPerfinesPrepagoNivles:str,
                 
                 PERIODO_PERFILES_DESDE :int,
                 PERIODO_PERFILES_HASTA :int,
                 NOMBRE_VARIABLES_PERFILES: str, 
                 PERIODO_SEGMENTACION_WEB:str,
 
                 pTablaMovilidad_General:str,
                 pTablaMovilidad_x_linea:str,
                 
                 Abonos_avg:list,
                 Abonos_sum:list,
                 Abonos_max:list,
                 Abonos_min:list,
                 Abonos_std:list,
                 
                 Perfiles_avg:list,
                 Perfiles_max:list,
                 
                 Movilidad_avg:list,
                 Movilidad_max:list,
                 
                 ABT_VARIABLES_NSS_NUM:str,
                 
                 
                 Prepago_avg:list,
                 Prepago_sum:list,
 
                
                 spark) -> None:
        
         self.MODELO=MODELO
         self.periodo=periodo
         self.TABLA_UNIVERSO=TABLA_UNIVERSO
         self.ELIMINAR_CORRELACIONES=ELIMINAR_CORRELACIONES
         self.CASTEAR_BIGINT=CASTEAR_BIGINT
         self.REDONDEAR_DECIMALES=REDONDEAR_DECIMALES 
         
         self.CAMPO_CLAVE=CAMPO_CLAVE
         self.AGRUPAR=AGRUPAR
         self.CAMPO_AGRUPAR=CAMPO_AGRUPAR
         
         self.DECIMALES_VARIABLES_NUMERICAS=DECIMALES_VARIABLES_NUMERICAS
         self.COTA_CORRELACIONES=COTA_CORRELACIONES
         self.REGISTROS_X_PARTICION=REGISTROS_X_PARTICION
         
         self.COTA_REPRESENTATIVIDAD=COTA_REPRESENTATIVIDAD
         
         self.pTablaPerfinesPospagoNivles=pTablaPerfinesPospagoNivles
         self.pTablaPerfinesPrepagoNivles=pTablaPerfinesPrepagoNivles
         
         self.PERIODO_PERFILES_DESDE=PERIODO_PERFILES_DESDE
         self.PERIODO_PERFILES_HASTA=PERIODO_PERFILES_HASTA
         self.NOMBRE_VARIABLES_PERFILES=NOMBRE_VARIABLES_PERFILES
         self.PERIODO_SEGMENTACION_WEB=PERIODO_SEGMENTACION_WEB
 
         self.pTablaMovilidad_General=pTablaMovilidad_General
         self.pTablaMovilidad_x_linea=pTablaMovilidad_x_linea
         
         self.Abonos_avg=Abonos_avg
         self.Abonos_sum=Abonos_sum
         self.Abonos_max=Abonos_max
         self.Abonos_min=Abonos_min
         self.Abonos_std=Abonos_std
         
         self.Perfiles_avg=Perfiles_avg
         self.Perfiles_max=Perfiles_max
         
         self.Movilidad_avg=Movilidad_avg
         self.Movilidad_max=Movilidad_max
         
         
         self.ABT_VARIABLES_NSS_NUM=ABT_VARIABLES_NSS_NUM
         self.Prepago_avg=Prepago_avg
         self.Prepago_sum=Prepago_sum
 
         
         self.spark=spark
         
    def DecimalToDouble(self, train_undersampled_df):
 
 
        numericCols = [c for c in train_undersampled_df.columns if c not in [self.CAMPO_CLAVE,'periodo', 'origin', 'label']]
        print("Num. numeric vars: " , len(numericCols))
 
        for c_name, c_type in train_undersampled_df.dtypes:
            if (c_type.find('decimal') >=0):
                train_undersampled_df = train_undersampled_df.withColumn(c_name, F.col(c_name).cast('double'))
 
        return train_undersampled_df             
    
    
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
    
    def Calcular_FTPrepago(self, pTabla_Salida):    
        try:
            self.spark.sql("DROP table sdb_datamining." + self.MODELO + 'ft_prepago_m_0')
        except:
            pass
 
        self.spark.sql("create table sdb_datamining." + self.MODELO + """ft_prepago_m_0 as 
                select b.*
                FROM """ + self.TABLA_UNIVERSO + """ a
                        inner join (select * 
                                    from data_lake_analytics.ft_prepago_m  
                                    where periodo = """ + str(self.periodo) + """
                                    ) b on a.linea = b.linea
                    """  )
 
        a = self.spark.sql("""select * 
                        from sdb_datamining.""" + self.MODELO + """ft_prepago_m_0
                        where periodo= """ + str(self.periodo) ).drop(*['fecha_alta_contrato',  'contrato', 'numero_documento']).fillna(0)
 
        a.write.mode('overwrite').format('parquet').saveAsTable("""sdb_datamining."""  + self.MODELO + "ft_prepago_m_1")
 
        self.CalcularABT("""sdb_datamining."""  + self.MODELO + "ft_prepago_m_1",  'sdb_datamining.' + self.MODELO + "ft_prepago_m")
 
        try:
            self.spark.sql("DROP table sdb_datamining." + self.MODELO + 'ft_prepago_m_0')
        except:
            pass
 
        try:
            self.spark.sql("DROP table sdb_datamining." + self.MODELO + 'ft_prepago_m_1')
        except:
            pass
 
        ft_prepago = self.spark.sql("select * from sdb_datamining." +  self.MODELO + "ft_prepago_m")
        print(ft_prepago.count())
        print(ft_prepago.columns)
 
 
        # Agrupo
 
        prepagos = self.spark.sql("select * from sdb_datamining." + self.MODELO + "ft_prepago_m limit 1").drop(*[self.CAMPO_CLAVE, self.CAMPO_AGRUPAR])
        prepagos_avg = self.AgruparCampos(self.Prepago_avg, prepagos.columns , 'avg', 'pospago')
        prepagos_sum = self.AgruparCampos(self.Prepago_sum, prepagos.columns , 'sum', 'pospago')
 
 
        try:
            self.spark.sql("drop table  " + pTabla_Salida )
        except:
            pass
 
        self.spark.sql("create table " + pTabla_Salida + """ as 
                    select a.""" + self.CAMPO_AGRUPAR +prepagos_avg + prepagos_sum + """
                    from  """ + self.TABLA_UNIVERSO + """ a,
                        sdb_datamining.""" + self.MODELO + """ft_prepago_m b
                    where a.linea = b.linea
                    group by a.""" + self.CAMPO_AGRUPAR)
 
        print(self.spark.sql("select count(1) from " + pTabla_Salida).show()) 
 
 
        try:
            self.spark.sql("DROP table sdb_datamining." + self.MODELO + 'ft_prepago_m')
        except:
            pass
        
    
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
        
 
    def calcularSum(self, pVar, pNombre):
      sql = ""
      sql2 = ""
      for h in pVar:
        if sql == "":
            sql += "sum(coalesce("+h+",0)"
        else: 
            sql += "+coalesce("+h+",0)"
        sql2 += ", sum(coalesce("+h+", 0)) as " + h + self.NOMBRE_VARIABLES_PERFILES
      sql += ") as total_" + pNombre + self.NOMBRE_VARIABLES_PERFILES
      return sql, sql2
    
    def calcularPorcentajes(self, pVar, pNombre):
      sql2 = ""
      for h in pVar:
        sql2 += ", round("+h+" / total_" + pNombre + self.NOMBRE_VARIABLES_PERFILES + ", 3) as " + h + "_porc"
      return sql2
    
    
    def calcularVersus(self, pVar, pNombre):
      sql = ""
      for h in pVar:
            for h2 in pVar:
                sql += ", round("+h+" / " + h2 + ", 3) as " + h + "_vs_" + h2
      return  sql
      
      
    def CrearABT_Perfiles_Mensual_Agrupados(self, pTabla_Salida):
        
        # Leo la tabla mensual
        perfiles_moviles = self.spark.sql(""" SELECT * FROM data_lake_analytics.stg_perfilesmovil_m  limit 10 """)
        sql_hits , sql_hits2 = self.calcularSum([ x for x in perfiles_moviles.columns if x.endswith('_hits')], 'hits')
        sql_mins, sql_mins2  = self.calcularSum([ x for x in perfiles_moviles.columns if x.endswith('_mins')], 'mins')
        sql_mbs, sql_mbs2    = self.calcularSum([ x for x in perfiles_moviles.columns if x.endswith('_mbs')], 'mbs')
        sql_apps, sql_apps2  = self.calcularSum([ x for x in perfiles_moviles.columns if x.endswith('_apps')], 'apps')
        sql_dias, sql_dias2  = self.calcularSum([ x for x in perfiles_moviles.columns if x.endswith('_dias')], 'dias')
        # Agrupo 
        try:
            query = " drop table sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_tx "
            self.spark.sql(query)
        except:
            pass
        query = "  create table sdb_datamining." + self.MODELO + """tmp_perfiles_moviles_tx as 
                select """ +  self.CAMPO_AGRUPAR + "," + \
                        sql_hits  + sql_hits2 +  "," + sql_mins +  sql_mins2 + "," + sql_mbs + sql_mbs2 + "," + sql_apps + sql_apps2 + "," + sql_dias + sql_dias2 + \
                """ from   """ + self.TABLA_UNIVERSO + """ a, 
                           data_lake_analytics.stg_perfilesmovil_m b
                where   b.periodo between """ + str(self.PERIODO_PERFILES_DESDE)  + """  and """ + str(self.PERIODO_PERFILES_HASTA) + """ 
                and     a.linea = b.linea
                group by a.""" + self.CAMPO_AGRUPAR
        train_undersampled_df = self.spark.sql(query)
        
        
        ######################################################################
        # sumo los dni que no encuentro en perfiles
        perfiles_moviles = self.spark.sql(" SELECT * FROM sdb_datamining." + self.MODELO + """tmp_perfiles_moviles_tx  limit 1 """).drop(self.CAMPO_AGRUPAR)
        sql = ""
        for i in perfiles_moviles.columns:
            sql += ', coalesce(' + i + ', 0) as ' + i
        train_undersampled_df = self.spark.sql("select  a." + self.CAMPO_AGRUPAR + sql + \
                                            " from (select distinct " + self.CAMPO_AGRUPAR + \
                                                    " from """ + self.TABLA_UNIVERSO + """) a
                                                    left join sdb_datamining.""" + self.MODELO + "tmp_perfiles_moviles_tx b on a." + self.CAMPO_AGRUPAR + " = b." + self.CAMPO_AGRUPAR
                                                    )
        train_undersampled_df.write.mode('overwrite').format('parquet').saveAsTable("sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_t")
        
        
        #######################################################################
        # porcentajes...
        columnas = self.VariablesRepresentativas("sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_t", self.COTA_REPRESENTATIVIDAD)
        print('<----------------- Porcentajes ----------------->')
        sql_hits3 = self.calcularPorcentajes([ x for x in columnas if x.endswith('_hits' + self.NOMBRE_VARIABLES_PERFILES)], 'hits')
        sql_mins3 = self.calcularPorcentajes([ x for x in columnas if x.endswith('_mins' + self.NOMBRE_VARIABLES_PERFILES)], 'mins')
        #sql_mbs3   = self.calcularPorcentajes([ x for x in columnas if x.endswith('_mbs'  + self.NOMBRE_VARIABLES_PERFILES)], 'mbs')
        #sql_apps3  = self.calcularPorcentajes([ x for x in columnas if x.endswith('_apps' + self.NOMBRE_VARIABLES_PERFILES)], 'apps')
        sql_dias3   = self.calcularPorcentajes([ x for x in columnas if x.endswith('_dias' + self.NOMBRE_VARIABLES_PERFILES)], 'dias')
        try:
            self.spark.sql( " drop table sdb_datamining."  + self.MODELO + "tmp_perfiles_moviles_t_0" )
        except:
            pass
        train_undersampled_df = self.spark.sql(""" 
                    select  a.""" + self.CAMPO_AGRUPAR + sql_hits3 + sql_mins3 + sql_dias3 +  \
                """ from   sdb_datamining.""" + self.MODELO + """tmp_perfiles_moviles_t a """ )
        train_undersampled_df.createOrReplaceTempView("train_undersampled_df")
        train_undersampled_df = self.ControlParticiones(train_undersampled_df, self.CAMPO_AGRUPAR, self.REGISTROS_X_PARTICION) 
        train_undersampled_df = self.CastBigInt(train_undersampled_df)
        train_undersampled_df = self.EliminarCorrelaciones(train_undersampled_df, self.COTA_CORRELACIONES)
        train_undersampled_df = self.ControlParticiones(train_undersampled_df, self.CAMPO_AGRUPAR, self.REGISTROS_X_PARTICION) 
        train_undersampled_df.fillna(0).write.mode('overwrite').format('parquet').saveAsTable("sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_t_0")
        print('columnas finales')
        print(train_undersampled_df.columns)
        
        
        #######################################################################
        # Busco eliminar las variables con poca representatividad (pocos clientes)
        print('<----------------- Versus ----------------->')
        sql_hits4 = self.calcularVersus([ x for x in columnas if x.endswith('_hits' + self.NOMBRE_VARIABLES_PERFILES)], 'hits')
        sql_mins4 = self.calcularVersus([ x for x in columnas if x.endswith('_mins' + self.NOMBRE_VARIABLES_PERFILES)], 'mins')
        #sql_mbs4   = self.calcularVersus([ x for x in columnas if x.endswith('_mbs'  + self.NOMBRE_VARIABLES_PERFILES)], 'mbs')
        #sql_apps4  = self.calcularVersus([ x for x in columnas if x.endswith('_apps' + self.NOMBRE_VARIABLES_PERFILES)], 'apps')
        sql_dias4  = self.calcularVersus([ x for x in columnas if x.endswith('_dias' + self.NOMBRE_VARIABLES_PERFILES)], 'dias')
        try:
            self.spark.sql( " drop table sdb_datamining."  + self.MODELO + "tmp_perfiles_moviles_t_1" )
        except:
            pass
        # Cruza con el parque, y los que no tienen trafico les pone las columnas como cero
        # sql_mins3 + sql_mbs3 + sql_mins4 + sql_mbs4 + sql_apps3 +  sql_apps4 +
        # create table sdb_datamining.""" + self.MODELO + """tmp_perfiles_moviles_t_1 as
        train_undersampled_df = self.spark.sql(""" 
                    select  a.""" + self.CAMPO_AGRUPAR +  sql_hits4  +   sql_dias4 + \
                    """ from    sdb_datamining.""" + self.MODELO + """tmp_perfiles_moviles_t a """ )
        train_undersampled_df.createOrReplaceTempView("train_undersampled_df")
        train_undersampled_df = self.ControlParticiones(train_undersampled_df, self.CAMPO_AGRUPAR,  self.REGISTROS_X_PARTICION) 
        train_undersampled_df = self.CastBigInt(train_undersampled_df)
        train_undersampled_df = self.EliminarCorrelaciones(train_undersampled_df, self.COTA_CORRELACIONES)
        train_undersampled_df = self.ControlParticiones(train_undersampled_df, self.CAMPO_AGRUPAR, self.REGISTROS_X_PARTICION) 
        train_undersampled_df.fillna(0).write.mode('overwrite').format('parquet').saveAsTable("sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_t_1")
        print('columnas finales')
        print(train_undersampled_df.columns)
        
        
        #########################################################
        print('<----------------- Total ----------------->')
        if 1 > 20 :
            train_undersampled_df = self.spark.sql("select distinct a." + self.CAMPO_AGRUPAR + sql + \
                                            " from """ + self.TABLA_UNIVERSO + """ a
                                                    left join sdb_datamining.""" + self.MODELO + "tmp_perfiles_moviles_tx b on a." + self.CAMPO_AGRUPAR + " = b." + self.CAMPO_AGRUPAR
                                                    ).select(*columnas)
                                                    
        train_undersampled_df = self.spark.sql("select " + self.CAMPO_AGRUPAR + sql + \
                                            " from sdb_datamining.""" + self.MODELO + "tmp_perfiles_moviles_t b"
                                                    ).select(*columnas)
        #train_undersampled_df = self.CastBigInt(train_undersampled_df)
        train_undersampled_df = self.EliminarCorrelaciones(train_undersampled_df, self.COTA_CORRELACIONES)
        train_undersampled_df = self.ControlParticiones(train_undersampled_df, self.CAMPO_AGRUPAR,  self.REGISTROS_X_PARTICION) 
        train_undersampled_df.fillna(0).write.mode('overwrite').format('parquet').saveAsTable("sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_t_2")
        print('columnas finales')
        print(train_undersampled_df.columns)
        
        ############################################################
        # Niveles
        
        print('Nivles')
        self.CalcularABT("""(select *, """ + str(self.periodo) + """ as Periodo 
                        from   sdb_datamining.""" + self.pTablaPerfinesPospagoNivles + """)""" , 'sdb_datamining.' +  self.MODELO + "tmp_perfiles_moviles_pospago_nivel")
        
        perfiles_pospago_niveles = self.spark.sql("select * from sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_pospago_nivel")
        print('Pospago ---> ' , perfiles_pospago_niveles.count())
        print(perfiles_pospago_niveles.columns)
    
    
        
        self.CalcularABT("""(select *, """ + str(self.periodo) + """ as Periodo 
                        from   sdb_datamining.""" + self.pTablaPerfinesPrepagoNivles + """)""" , 'sdb_datamining.' +  self.MODELO + "tmp_perfiles_moviles_prepago_nivel")
        
        perfiles_prepago_niveles = self.spark.sql("select * from sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_prepago_nivel")
        print('Prepago ---> ' ,perfiles_prepago_niveles.count())
        print(perfiles_prepago_niveles.columns)
        
        ##########################################################
        # Agrupo
        
        perfiles_pospago_niveles = self.spark.sql("select * from sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_pospago_nivel limit 1").drop(*[self.CAMPO_CLAVE, self.CAMPO_AGRUPAR])
        perfiles_pospago_max = self.AgruparCampos(self.Perfiles_max, [x for x in perfiles_pospago_niveles.columns if x.endswith('nivel') ], 'max', 'pospago')
        
        perfiles_prepago_niveles = self.spark.sql("select * from sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_prepago_nivel limit 1" ).drop(*[self.CAMPO_CLAVE, self.CAMPO_AGRUPAR])
        perfiles_prepago_max = self.AgruparCampos(self.Perfiles_max, [x for x in perfiles_prepago_niveles.columns if x.endswith('nivel') ], 'max', 'prepago')
        
        
        if self.AGRUPAR == True:
            
            perfiles_pospago_promedios = self.AgruparCampos(self.Perfiles_avg, [x for x in perfiles_pospago_niveles.columns if x.endswith('nivel') ], 'avg', 'pospago')
            perfiles_prepago_promedios = self.AgruparCampos(self.Perfiles_avg, [x for x in perfiles_prepago_niveles.columns if x.endswith('nivel') ], 'avg', 'prepago')
            
        else:
        
            perfiles_pospago_promedios = ""
            perfiles_prepago_promedios = ""
            
        
        linea = ', count(distinct a.linea) as cantidad_lineas'
        try:
            self.spark.sql("drop table sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_pospago_nivel_agrup")
        except:
            pass
        
        self.spark.sql("create table sdb_datamining." + self.MODELO + """tmp_perfiles_moviles_pospago_nivel_agrup as 
                    select a.""" + self.CAMPO_AGRUPAR + perfiles_pospago_promedios + perfiles_pospago_max + linea  + """
                    from """ + self.TABLA_UNIVERSO + """ a,
                          sdb_datamining.""" + self.MODELO + """tmp_perfiles_moviles_pospago_nivel b
                    where a.linea = b.linea
                    group by a.""" + self.CAMPO_AGRUPAR)
           
           

    
        #############################################################
        # Junto
        
        try:
            self.spark.sql("drop table sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_prepago_nivel_agrup")
        except:
            pass
        
        self.spark.sql("create table sdb_datamining." + self.MODELO + """tmp_perfiles_moviles_prepago_nivel_agrup as 
                    select a.""" + self.CAMPO_AGRUPAR + perfiles_prepago_promedios + perfiles_prepago_max  + """
                    from  """ + self.TABLA_UNIVERSO + """ a,
                          sdb_datamining.""" + self.MODELO + """tmp_perfiles_moviles_prepago_nivel b
                    where a.linea = b.linea
                    group by a.""" + self.CAMPO_AGRUPAR)
                    
        
        # Junto todas las de perfiles
        perfiles_t0 = self.spark.sql("select * from sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_t_0") # Ya agrupado
        perfiles_t1 = self.spark.sql("select * from sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_t_1") # Ya agrupado
        perfiles_t2 = self.spark.sql("select * from sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_t_2") # Ya agrupado
        perfiles_pospago_niveles = self.spark.sql("select * from sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_pospago_nivel_agrup") # ya agrupado
        perfiles_prepago_niveles = self.spark.sql("select * from sdb_datamining." + self.MODELO + "tmp_perfiles_moviles_prepago_nivel_agrup") # ya agrupado
        
        abt_perfiles_dni = perfiles_t0.join(perfiles_t1, [self.CAMPO_AGRUPAR], 'inner').join(perfiles_t2, [self.CAMPO_AGRUPAR], 'inner')\
                                .join(perfiles_prepago_niveles, [self.CAMPO_AGRUPAR], 'inner').join(perfiles_pospago_niveles, [self.CAMPO_AGRUPAR], 'inner')
        
        abt_perfiles_dni.write.mode('overwrite').format('parquet').saveAsTable(pTabla_Salida)
    
    
        try:
            self.spark.sql(""" drop table sdb_datamining.""" + self.MODELO + """tmp_perfiles_moviles_t""")
        except:
            pass
        
        try:
            self.spark.sql( " drop table sdb_datamining."  + self.MODELO + "tmp_perfiles_moviles_t_0" )
        except:
            pass
        
        try:
            self.spark.sql( " drop table sdb_datamining."  + self.MODELO + "tmp_perfiles_moviles_t_1" )
        except:
            pass
        
        try:
            self.spark.sql( " drop table sdb_datamining."  + self.MODELO + "tmp_perfiles_moviles_t_2" )
        except:
            pass
              
    def CalcularMovilidad_Agrupada(self, AGRUPAR_POR, pTablaSalida):             
    
        # Cruzo con tola la info que necesito             
        pTablaSalida = " sdb_datamining." + self.MODELO + "_movilidad_v_prov_loc "
        self.CalcularVariablesGeolocalizacion(AGRUPAR_POR, self.pTablaMovilidad_General, pTablaSalida)
        
        # Cruzo con el universo
        
        a = self.spark.sql(" select * from sdb_datamining." + self.MODELO + "_movilidad_v_prov_loc limit 1").drop(self.CAMPO_CLAVE, 'periodo', 'provincia', 'localidad_barrio')
        ABT_VARIABLES = str(a.columns).replace("'", "").replace("[", "").replace("]", "")
        
        a = self.spark.sql(""" select a.linea,  """ + ABT_VARIABLES + """
                   FROM  """ + self.TABLA_UNIVERSO + """ a
                        left join (      
                                    select linea,  """ + ABT_VARIABLES + """
                                        from     """ + self.pTablaMovilidad_x_linea + """  a,
                                                
                                                sdb_datamining.""" + self.MODELO + """_movilidad_v_prov_loc b 
                                    where a.vive_provincia = b.provincia and a.vive_localidad_barrio = b.localidad_barrio
                                    ) b on a.linea = b.linea
                   
                        """  ).fillna(0)
                        
        
        a.write.mode('overwrite').format('parquet').saveAsTable("sdb_datamining." + self.MODELO + "_linea_movilidad_v_prov_loc_l")
        
        # Formateo la tabla
        
        self.CalcularABT(""" (select *, """ + str(self.periodo) + """ as Periodo 
                        from  sdb_datamining.""" + self.MODELO + "_linea_movilidad_v_prov_loc_l ) ", 'sdb_datamining.' +  self.MODELO + "_tmp_movilidad_v_prov_loc_lx")
        
        movilidad_v_prov_loc = self.spark.sql("select * from sdb_datamining." + self.MODELO + "_tmp_movilidad_v_prov_loc_lx")
        print(movilidad_v_prov_loc.count())
        print(movilidad_v_prov_loc.columns)
        
        # Agrupar
        # Pongo todas las columnas que se generaron
        a = self.spark.sql(" select * from sdb_datamining." + self.MODELO + "_movilidad_v_prov_loc limit 1").drop(self.CAMPO_CLAVE, 'periodo', 'provincia', 'localidad_barrio')
        
        movilidad_provloc = self.spark.sql("select * from sdb_datamining." + self.MODELO + "_tmp_movilidad_v_prov_loc_lx limit 1").drop(*[self.CAMPO_CLAVE, self.CAMPO_AGRUPAR])
        movilidad_provloc_max = self.AgruparCampos(a.columns, movilidad_provloc.columns , 'max', 'pospago')
        
        if self.AGRUPAR == True:
            movilidad_provloc_promedios = self.AgruparCampos(a.columns, movilidad_provloc.columns , 'avg', 'pospago')
        else:
            movilidad_provloc_promedios = ""
        
        try:
            self.spark.sql("drop table  " + pTablaSalida )
        except:
            pass
            
        self.spark.sql("create table " + pTablaSalida + """ as 
                    select a.""" + self.CAMPO_AGRUPAR + movilidad_provloc_max + movilidad_provloc_promedios   + """
                    from  """ + self.TABLA_UNIVERSO + """ a,
                          sdb_datamining.""" + self.MODELO + """_tmp_movilidad_v_prov_loc_lx b
                    where a.linea = b.linea
                    group by a.""" + self.CAMPO_AGRUPAR)
                    
        print(self.spark.sql("select count(1) from sdb_datamining." + self.MODELO + """_tmp_movilidad_v_prov_loc_lx_agrup""").show()) 
    
        
        try:
            self.spark.sql("DROP table sdb_datamining." + self.MODELO + '_linea_movilidad_v_prov_loc_l')
        except:
            pass
        
        try:
            self.spark.sql("drop table sdb_datamining." + self.MODELO + "_tmp_movilidad_v_prov_loc_lx")
        except:
            pass
        
    
    def CreateTmpNosis(self, pTabla_Salida):
        
        try:
            query = " drop table sdb_datamining." + self.MODELO + "_tmp_nosis "
            self.spark.sql(query)
    
        except:
            pass
        
        query = "create table sdb_datamining." + self.MODELO + """_tmp_nosis as
            select distinct a.dni, """ + self.ABT_VARIABLES_NSS_NUM + """
            from    (select distinct dni 
                    from sdb_datamining.""" + self.MODELO + """_universo) a
            left join (select distinct cast(trim(substring(doc_nro, 3, 8)) as string) as dni, """ + self.ABT_VARIABLES_NSS_NUM + """ 
                        from sdb_datamining.cdas_nss_validados_external) b 
                on a.dni = b.dni"""
        self.spark.sql(query)
        
        columnas = self.VariablesRepresentativas("sdb_datamining." + self.MODELO + "_tmp_nosis", self.COTA_REPRESENTATIVIDAD)
    
        ####################################################################
        # 1.3. Corregir Numeros, Eliminar Correlaciones y Particiones
        ####################################################################
        
        train_undersampled_df = self.spark.sql("select * from sdb_datamining." + self.MODELO + "_tmp_nosis").select(*columnas)
        
        train_undersampled_df = self.DecimalToDouble(train_undersampled_df).fillna(0)
        if self.CASTEAR_BIGINT == True:
            train_undersampled_df = self.CastBigInt(train_undersampled_df)
            
        if self.REDONDEAR_DECIMALES == True:
            train_undersampled_df = self.RedondearDecimales(train_undersampled_df, self.DECIMALES_VARIABLES_NUMERICAS)
            print('Redondeo......')
            
        if self.ELIMINAR_CORRELACIONES == True:
            train_undersampled_df = self.EliminarCorrelaciones(train_undersampled_df, self.COTA_CORRELACIONES)
    
        train_undersampled_df = self.ControlParticiones(train_undersampled_df, self.CAMPO_AGRUPAR , self.REGISTROS_X_PARTICION) 
        train_undersampled_df.write.mode('overwrite').format('parquet').saveAsTable(pTabla_Salida  )
    
        print(train_undersampled_df.count())
        
        try:
            self.spark.sql("drop table sdb_datamining." + self.MODELO + """_tmp_nosis """)
        except:
            pass
    
    
    def CalcularVariablesGeolocalizacion(self, AGRUPAR_POR, pTablaMovilidad, pTablaSalida):    
    
        a = self.spark.sql("select * from sdb_datamining.cluster_perfiles_prepago limit 1")
        variables_perfiles = ""
        variables_perfiles_avg = ""
        for i in [x for x in a.columns if (x != 'linea') & (x != 'periodo')]:
                variables_perfiles += ', coalesce(' + i + ', 0) as ' + i
                if i != 'prediction':
                    variables_perfiles_avg += ', sum(case when coalesce(' + i + ', 0) = 0 then 1 else 0 end) as ' + i + '_0_q'\
                                              ', sum(case when coalesce(' + i + ', 0) = 1 then 1 else 0 end) as ' + i + '_1_q'\
                                              ', sum(case when coalesce(' + i + ', 0) = 2 then 1 else 0 end) as ' + i + '_2_q'\
                                              ', sum(case when coalesce(' + i + ', 0) = 3 then 1 else 0 end) as ' + i + '_3_q'\
                                              ', sum(case when coalesce(' + i + ', 0) = 4 then 1 else 0 end) as ' + i + '_4_q'\
                                              ', sum(case when coalesce(' + i + ', 0) = 5 then 1 else 0 end) as ' + i + '_5_q'\
                                              ', sum(case when coalesce(' + i + ', 0) = 6 then 1 else 0 end) as ' + i + '_6_q'
        
        
        try:
            self.spark.sql("drop table sdb_datamining." + self.MODELO + """_universo_v """)
        except:
            pass
        self.spark.sql("create table sdb_datamining." + self.MODELO + """_universo_v as 
                   select   trim(substr(tipo_nro_doc_norm, 3, 10)) as dni, 
                            a.id_parque as linea, 
                            
                            /*Parque Convergente */
                            a.f_pospago, a.f_prepago, a.flg_recarga, 
                            
                            /* Vive Trabaja*/
                            b.tipo, b.actividad_probable, b.provincia, b.localidad_barrio, b.celda
                            
                            /*Perfiles*/
                            """ + variables_perfiles + """,
                            
                            /*Nosis */
                            edad, genero, nse, bco_cant,
                            bcra_sit_vg, telcos_mor_cant, tc_cant,
                            es_empleado, es_jubilado,
                            es_pensionado, es_monotrib,
                            es_autonomo 
                            
                   FROM stg_repo_convergente.vc_vision_conv_unificado_aa a
                            inner join (select a.* 
                                        from """ + self.pTablaMovilidad_General + """ a
                                            inner join (select linea, max(celda) celda
                                                        from """ + self.pTablaMovilidad_General + """
                                                        where actividad_probable = 'Probablemente vive'
                                                        and   tipo = 1
                                                        and   iso_3166_1 = 'AR'
                                                        group by linea
                                                        )  b  on a.linea = b.linea and a.celda = b.celda
                                        where actividad_probable = 'Probablemente vive'
                                        and   tipo = 1
                                        and   iso_3166_1 = 'AR') b on a.id_parque = cast(b.linea as double)
                                        
                                        
                            left join (select linea """ + variables_perfiles + """
                                        from sdb_datamining.cluster_perfiles_pospago 
                                        where  periodo = '""" + self.PERIODO_SEGMENTACION_WEB + """'
                                        union
                                        select linea """ + variables_perfiles + """
                                        from sdb_datamining.cluster_perfiles_prepago
                                        where  periodo = '""" + self.PERIODO_SEGMENTACION_WEB + """'
        
                                        ) c on a.id_parque = c.linea
        
                            left join (select *
                                        from data_lake_analytics.stg_parquemovil_m 
                                        where periodo= """ + str(self.periodo) + """
                                        ) d on a.id_parque = d.linea
        
                    where  a.periodo= """ + str(self.periodo) + """
                    and    a.empresa = 'TP' 
                    """  )
    
        try:
            self.spark.sql("drop table sdb_datamining." + self.MODELO + """_universo_v_prov_loc_0 """)
        except:
            pass
        
        self.spark.sql("create table sdb_datamining." + self.MODELO + """_universo_v_prov_loc_0 as 
        select """ + AGRUPAR_POR + """, 
               
                count(1) as lineas_con_vt,
                /*count(distinct dni) as dni_q,*/
                sum(a.f_pospago) as f_pospago_q, 
                sum(a.f_prepago) as f_prepago_q, 
                sum(a.flg_recarga ) as flg_recarga_q,
                
                sum(case when genero = 'F' then 1 else 0 end) as genero_F_q,
                sum(case when genero = 'M' then 1 else 0 end) as genero_M_q,
                sum(case when genero = 'I' then 1 else 0 end) as genero_I_q,
                sum(case when genero is null then 1 else 0 end) as genero_null_q,
                
                sum(case when nse = 'A' then 1 else 0 end) as nse_A_q,
                sum(case when nse = 'B' then 1 else 0 end) as nse_B_q,
                sum(case when nse in ('C1', 'C2', 'C3') then 1 else 0 end) as nse_C_q,
                sum(case when nse in ('D1', 'D2', 'D3') then 1 else 0 end) as nse_D_q,
                sum(case when nse = 'NC' then 1 else 0 end) as nse_NC_q,
                sum(case when nse is null then 1 else 0 end) as nse_null_q,
                
                sum(case when prediction = 0 then 1 else 0 end ) as Segmentacion_Uso_Web_0_q,
                sum(case when prediction = 1 then 1 else 0 end ) as Segmentacion_Uso_Web_1_q,
                sum(case when prediction = 2 then 1 else 0 end ) as Segmentacion_Uso_Web_2_q,
                sum(case when prediction = 3 then 1 else 0 end ) as Segmentacion_Uso_Web_3_q,
                sum(case when prediction = 4 then 1 else 0 end ) as Segmentacion_Uso_Web_4_q,
                sum(case when prediction = 5 then 1 else 0 end ) as Segmentacion_Uso_Web_5_q,
                
                sum(case when edad < 20 then 1 else 0 end) as edad_menos20_q,
                sum(case when edad between 21 and 30 then 1 else 0 end) as edad_21_30_q,
                sum(case when edad between 31 and 40 then 1 else 0 end) as edad_31_40_q,
                sum(case when edad between 41 and 50 then 1 else 0 end) as edad_41_50_q,
                sum(case when edad between 51 and 60 then 1 else 0 end) as edad_51_60_q,
                sum(case when edad between 61 and 70 then 1 else 0 end) as edad_61_70_q,
                sum(case when edad = 71  then 1 else 0 end) as edad_mas70_q,
                
                sum(case when bco_cant > 0 then 1 else 0 end) as lineas_bco_cant_q,
               avg(case when bco_cant > 0 then bco_cant end) as bco_cant_avg,
                
                sum(case when bcra_sit_vg > 0 then 1 else 0 end) as lineas_con_bcra_sit_vg_mas0_q, 
                sum(case when bcra_sit_vg >= 3 then 1 else 0 end) as lineas_con_bcra_sit_vg_mas2_q, 
                
                sum(case when telcos_mor_cant > 0 then 1 else 0 end) as lineas_con_telcos_mor_cant_q, 
                sum(tc_cant) as tc_cant_q,
                sum(case when tc_cant > 0 then 1 else 0 end) as lineas_con_tc_cant_q,
                avg(case when tc_cant > 0 then tc_cant end) as tc_avg,
                
                sum(es_empleado) as es_empleado_q, 
                sum(es_jubilado) as es_jubilado_q,
                sum(es_pensionado) as es_pensionado_q, 
                sum(es_monotrib) as es_monotrib_q,
                sum(es_autonomo) as es_autonomo_q
                
                """ + variables_perfiles_avg + """
                
        from sdb_datamining." + self.MODELO + "_universo_v a
        group by """ + AGRUPAR_POR )
        
        # Calculo Porcentajes
        
        a = self.spark.sql("select * from sdb_datamining." + self.MODELO + "_universo_v_prov_loc_0  limit 1")
        variables_q = ""
        for i in [x for x in a.columns if x.endswith('_q')]:
                variables_q += ', cast( round((' + i + ' / lineas_con_vt ) * 100) as smallint) as ' + i + '_porc'
                
        variables_no_q = list([x for x in a.columns if x.endswith('_q') == 0])
        variables_no_q = str(variables_no_q).replace("'", "").replace("[", "").replace("]", "")
        variables_no_q
        
        #####################################
    
        try:
            self.spark.sql("drop table  """ + pTablaSalida )
        except:
            pass
        
        self.spark.sql("create table """ + pTablaSalida + """ as 
                 select """ + variables_no_q + variables_q + \
                 " from sdb_datamining." + self.MODELO + """_universo_v_prov_loc_0 """)
                 
        print('count...', self.spark.sql("select count(1) from " + pTablaSalida ).show())
    
        print(pTablaSalida)
        
        #####################################
        try:
            self.spark.sql("drop table sdb_datamining." + self.MODELO + """_universo_v """)
        except:
            pass
        
        try:
            self.spark.sql("drop table sdb_datamining." + self.MODELO + """_universo_v_prov_loc_0 """)
        except:
            pass
        
             
    def CalcularMovilidad_SinAgrupar(self, pTabla_Salida):             
        
        # Esto se calcula en GCP
        
        self.CalcularABT("""(select *, """ + str(self.periodo) + """ as Periodo 
                        from   """ + self.pTablaMovilidad_x_linea + """ )""" , 'sdb_datamining.' +  self.MODELO + "tmp_movilidad_linea")
        
        movilidad_linea = self.spark.sql("select * from sdb_datamining." + self.MODELO + "tmp_movilidad_linea")
        print(movilidad_linea.count())
        print(movilidad_linea.columns)
        
        movilidad = self.spark.sql("select * from sdb_datamining." + self.MODELO + "tmp_movilidad_linea limit 1" ).drop(*[self.CAMPO_CLAVE, self.CAMPO_AGRUPAR])
        
        movilidad_max = self.AgruparCampos(self.Movilidad_max, movilidad.columns , 'max', '')
        
        if self.AGRUPAR == True:
            movilidad_avg = self.AgruparCampos(self.Movilidad_avg, movilidad.columns , 'avg', '')
        else:
            movilidad_avg = ""
        
        try:
            self.spark.sql("drop table " + pTabla_Salida )
        except:
            pass
        
        self.spark.sql("create table " + pTabla_Salida  + """ as 
                select a.""" + self.CAMPO_AGRUPAR + movilidad_avg + movilidad_max  + """
                from  """ + self.TABLA_UNIVERSO + """ a,
                      sdb_datamining.""" + self.MODELO + """tmp_movilidad_linea b
                where a.linea = b.linea
                group by a.""" + self.CAMPO_AGRUPAR)
                
    
        print(self.spark.sql("select count(1) from " + pTabla_Salida ).show(1))
        
    
    def Calcular_FTAbono(self, pTabla_Salida):
    
        try:
            self.spark.sql("DROP table sdb_datamining." + self.MODELO + 'ft_abonos_m_0')
        except:
            pass
        
        self.spark.sql("create table sdb_datamining." + self.MODELO + """ft_abonos_m_0 as 
                   select b.*
                   FROM """ + self.TABLA_UNIVERSO + """ a
                        inner join (select * 
                                    from data_lake_analytics.ft_abonos_m  
                                    where periodo = """ + str(self.periodo) + """
                                    ) b on a.linea = b.linea
                    """  )
                    
        a = self.spark.sql("""select * 
                        from sdb_datamining.""" + self.MODELO + """ft_abonos_m_0
                        where periodo= """ + str(self.periodo) ).drop(*['cbs_cust_id', 'cliente', 'numerodocumento']).fillna(0)
        
        a.write.mode('overwrite').format('parquet').saveAsTable("""sdb_datamining."""  + self.MODELO + "ft_abonos_m_1")
        
        self.CalcularABT("""sdb_datamining."""  + self.MODELO + "ft_abonos_m_1",  'sdb_datamining.' +  self.MODELO + "ft_abonos_m")
        
        try:
            self.spark.sql("DROP table sdb_datamining." + self.MODELO + 'ft_abonos_m_0')
        except:
            pass
        
        try:
            self.spark.sql("DROP table sdb_datamining." + self.MODELO + 'ft_abonos_m_1')
        except:
            pass
        
        
        ft_abonos = self.spark.sql("select * from sdb_datamining." +  self.MODELO + "ft_abonos_m")
        print(ft_abonos.count())
        print(ft_abonos.columns)
        
        
        # Agrupo
        
        
        
        abonos = self.spark.sql("select * from sdb_datamining." + self.MODELO + "ft_abonos_m limit 1").drop(*[self.CAMPO_CLAVE, self.CAMPO_AGRUPAR])
        abonos_avg = self.AgruparCampos(self.Abonos_avg, abonos.columns , 'avg', 'pospago')
        abonos_sum = self.AgruparCampos(self.Abonos_sum, abonos.columns , 'sum', 'pospago')
        abonos_max = self.AgruparCampos(self.Abonos_max, abonos.columns , 'max', 'pospago')
        abonos_min = self.AgruparCampos(self.Abonos_min, abonos.columns , 'min', 'pospago')
        abonos_std = self.AgruparCampos(self.Abonos_std, abonos.columns , 'STDDEV', 'pospago')
        
        
        try:
            self.spark.sql("drop table  " + pTabla_Salida )
        except:
            pass
            
        self.spark.sql("create table " + pTabla_Salida + """ as 
                    select a.""" + self.CAMPO_AGRUPAR +abonos_avg + abonos_sum + abonos_max + abonos_min + abonos_std  + """
                    from  """ + self.TABLA_UNIVERSO + """ a,
                          sdb_datamining.""" + self.MODELO + """ft_abonos_m b
                    where a.linea = b.linea
                    group by a.""" + self.CAMPO_AGRUPAR)
                    
        print(self.spark.sql("select count(1) from " + pTabla_Salida).show()) 
    
    
    def AgruparCampos(self, Variables_Agrupar, Variables_Input, function, prefijo):
      sql = ""
      for h in Variables_Input:
        if h in Variables_Agrupar:
            a = function + "(coalesce("+h+",0)) as " + h + '_' + prefijo + '_' + function  # ver aca si estan bien los coalesce!!!!!!!!!!!!!!!! 
            sql += ", " + a
                
      return sql
      
    def calcularRepresentatividad(self, pVar):
      sql = ""
      sql2 = ""
      for h in pVar:
        if sql == "":
            sql2 += " sum(case when "+h+" != 0 then 1 else 0 end) as " + h 
            sql +=   h + "/ total_Distinct * 100 as " +h 
       
        else:
            sql2 += ", sum(case when "+h+" != 0 then 1 else 0 end) as " + h 
            sql +=  "," + h + "/ total_Distinct * 100 as " +h 
       
      sql2 += ", sum(1) as total_distinct  "
      return sql2, sql
      
      
      
    def VariablesRepresentativas(self, pTabla, pCota):
        columnas = self.spark.sql(" select * from " + pTabla + " limit 1""").columns
            
        sql_r, sql_r1 = self.calcularRepresentatividad([ x for x in columnas ] )
        
        representarividad = self.spark.sql(" select " + sql_r1 + \
                            "  from ( select " + sql_r  + \
                                     " from  " + pTabla + """ ) """).toPandas().T.reset_index()
        representarividad.columns = ['feature', 'porc']
        
        representarividad = representarividad[representarividad.porc > pCota]
    
        return list(representarividad['feature'])
    
    
    def CalcularABT(self, pTABLA_ABT, pTABLA_SALIDA):
    
        a = self.spark.sql(" select * from " + pTABLA_ABT + " limit 1").drop(*['contrato_key', 'idcuentafacturacion', 'subscription_id', 'periodo', 'linea'])
        ABT_VARIABLES = str(a.columns).replace("'", "").replace("[", "").replace("]", "")
        ABT_VARIABLES
    
        try:
            self.spark.sql("DROP table sdb_datamining." + self.MODELO + '_ABT_1')
        except:
            pass
        
        self.spark.sql("create table sdb_datamining." + self.MODELO + """_ABT_1 as 
                   select a.linea, """ + ABT_VARIABLES + """
                   FROM """ + self.TABLA_UNIVERSO + """ a
                        left join (select * 
                                    from """ + pTABLA_ABT + """  
                                    where periodo = """ + str(self.periodo) + """
                                    ) b on a.linea = b.linea
                    """  )
                    
        
        columnas = self.VariablesRepresentativas("sdb_datamining." + self.MODELO + "_ABT_1", self.COTA_REPRESENTATIVIDAD)
    
        ####################################################################
        # 1.3. Corregir Numeros, Eliminar Correlaciones y Particiones
        ####################################################################
        
        train_undersampled_df = self.spark.sql("select * from sdb_datamining." + self.MODELO + "_ABT_1").select(*columnas)
        
        train_undersampled_df = self.DecimalToDouble(train_undersampled_df).fillna(0)
        if self.CASTEAR_BIGINT == True:
            train_undersampled_df = self.CastBigInt(train_undersampled_df)
            
        if self.REDONDEAR_DECIMALES == True:
            train_undersampled_df = self.RedondearDecimales(train_undersampled_df, self.DECIMALES_VARIABLES_NUMERICAS)
            print('Redondeo......')
            
        if self.ELIMINAR_CORRELACIONES == True:
            train_undersampled_df = self.EliminarCorrelaciones(train_undersampled_df, self.COTA_CORRELACIONES)
    
        train_undersampled_df = self.ControlParticiones(train_undersampled_df, self.CAMPO_CLAVE, self.REGISTROS_X_PARTICION) 
        train_undersampled_df.write.mode('overwrite').format('parquet').saveAsTable(pTABLA_SALIDA)
        
        try:
            self.spark.sql("drop table sdb_datamining." + self.MODELO + """_ABT_1""")
        except:
            pass

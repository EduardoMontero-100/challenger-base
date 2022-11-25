class Challenger():
    def __init__(self) -> None:
        pass
    
    
    
    def BorrarTablasTemporales():

    try:
        spark.sql(' drop table sdb_datamining.' + modelo + '_0' )
    except:
        pass

    try:
        spark.sql(' drop table sdb_datamining.' + modelo + '_1' )
    except:
        pass

    try:
        spark.sql(' drop table sdb_datamining.' + modelo + '_2' )
    except:
        pass


    try:
        spark.sql(' drop table sdb_datamining.' + modelo + '_testing' )
    except:
        pass
    
    
    
    # Estas 2 son las que hay que grabar....
    
    
    try:
        spark.sql(' drop table sdb_datamining.' + modelo + '_metricas' )
    except:
        pass
    
    
    try:
        spark.sql(' drop table sdb_datamining.' + modelo + '_feature_importance' )
    except:
        pass
    
    
    try:
        spark.sql(' drop table sdb_datamining.' + modelo + '_feature_importance_rank' )
    except:
        pass
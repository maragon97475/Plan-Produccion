

IF OBJECT_ID('tempdb..#discharge_line_adherence') IS NOT NULL DROP TABLE #discharge_line_adherence
IF OBJECT_ID('tempdb..#historia_mareas') IS NOT NULL DROP TABLE #historia_mareas
IF OBJECT_ID('tempdb..#mareas_acodere') IS NOT NULL DROP TABLE #mareas_acodere
IF OBJECT_ID('tempdb..#MAREASCLASIF') IS NOT NULL DROP TABLE #MAREASCLASIF
IF OBJECT_ID('tempdb..#ULTREC') IS NOT NULL DROP TABLE #ULTREC

--TEMPORAL MAREAS WEBSERVICE
SELECT marea_id,boat_name,discharge_plant_arrival_date,eta,declared_ton,actual_ton_discharge,
discharge_plant_name=CASE WHEN discharge_plant_name='CHICAMA' THEN 'MALABRIGO' 
						WHEN discharge_plant_name='PISCO' OR discharge_plant_name='PISCO NORTE' THEN 'PISCO SUR' 
						WHEN discharge_plant_name='MOLLENDO' THEN  'MATARANI'
						ELSE discharge_plant_name END,
discharge_chata_name=CASE WHEN discharge_chata_name='CHATA EXABA' THEN 'CHATA EX-ABA' 
						--AGREGADO
						WHEN discharge_chata_name='ALCATRAZ' THEN 'CHATA ALCATRAZ'
						ELSE discharge_chata_name END, 
discharge_poza_1,discharge_poza_2,discharge_poza_3, production_date,discharge_start_date
INTO #historia_mareas
FROM MareasWebService WITH(NOLOCK)
--WHERE CONVERT(VARCHAR(6),production_date,112)>=202111

--TEMPORAL MAREAS ACODERE
SELECT MA.marea_id,
MA.linea_descarga_acodere,
MA.motive_linea_descarga,MA.acodera_chata,MA.inicio_succion,
MA.termino_succion,MA.desacodera_chata,SPM.acodera_inicio_succion,MA.motive_acodere_inicio,SPM.fin_succion_desacodere,
MA.motive_acodere_fin,SPM.acoderar,MA.motive_acodere,MA.poza_1,MA.poza_2,MA.poza_3,MA.motive_poza,MA.timestamp,MA.by_user,
MA.orden_descarga,MA.discharge_start_date,
IIF(MA.chata_descarga_acodere='EX','EX-ABA',MA.chata_descarga_acodere) chata_descarga_acodere,
MA.motive_acodere_order,MA.poza_4,
SPM.acodere_orden_descarga as orden_asignado,OAO1.des_category as motivo_acodere_orden, OAO2.des_category as motivo_acodere_inicio,
OAO3.des_category as motivo_acodere_fin,OAO4.des_category as motivo_acodere,OAO5.des_category as motivo_linea_descarga,
OAO6.des_category as motivo_poza
INTO #mareas_acodere
FROM MareasAcodere MA WITH(NOLOCK)
LEFT JOIN OrderAdherenceOptions OAO1 WITH(NOLOCK) ON OAO1.id=MA.motive_acodere_order
LEFT JOIN OrderAdherenceOptions OAO2 WITH(NOLOCK) ON OAO2.id=MA.motive_acodere_inicio
LEFT JOIN OrderAdherenceOptions OAO3 WITH(NOLOCK) ON OAO3.id=MA.motive_acodere_fin
LEFT JOIN OrderAdherenceOptions OAO4 WITH(NOLOCK) ON OAO4.id=MA.motive_acodere
LEFT JOIN OrderAdherenceOptions OAO5 WITH(NOLOCK) ON OAO5.id=MA.motive_linea_descarga
LEFT JOIN OrderAdherenceOptions OAO6 WITH(NOLOCK) ON OAO6.id=MA.motive_poza
LEFT JOIN SPMareasJoinedRecomsHistorico SPM WITH(NOLOCK) ON SPM.marea_id=MA.marea_id
--WHERE CONVERT(VARCHAR(6),timestamp,112)>=202111

--SELECT * FROM #MAREASCLASIF
--CLASIFICACION INTERNA MAREAS
SELECT marea_id,Clasificacion,FechaFinalRec
INTO #MAREASCLASIF
FROM(
	SELECT marea_id,Clasificacion,FechaRecomendacion,MIN(FechaRecomendacion) OVER(PARTITION BY marea_id)FechaFinalRec
	FROM(
		--MAREAS CON RECOMENDACION ANTES DE FECHA ACODERE (CASO ESTANDAR)
		SELECT MA.marea_id,MAX(RHIST.last_modification)FechaRecomendacion,'Rec. antes Acodere' Clasificacion
		FROM #mareas_acodere MA
		INNER JOIN RetornoRecomendacionHistorico RHIST WITH(NOLOCK) ON RHIST.marea_id=MA.marea_id
		WHERE RHIST.last_modification<MA.acodera_chata
		GROUP BY MA.marea_id
		UNION ALL
		--MAREAS CON RECOMENDACION DESPUES FECHA ACODERE (PARA QUE COINCIDA CON MIO ULTIMA REC)
		SELECT MA.marea_id,MAX(RHIST.last_modification)FechaRecomendacion,'Rec. después acodere' Clasificacion
		FROM #mareas_acodere MA
		INNER JOIN RetornoRecomendacionHistorico RHIST WITH(NOLOCK) ON RHIST.marea_id=MA.marea_id
		WHERE RHIST.last_modification>MA.acodera_chata
		GROUP BY MA.marea_id
		UNION ALL
		--MAREAS SIN FECHA ACODERE CHATA
		SELECT MA.marea_id,MAX(RHIST.last_modification) FechaRecomendacion,'Sin acodere chata' Clasificacion
		FROM #mareas_acodere MA
		INNER JOIN #historia_mareas HM ON HM.marea_id=MA.marea_id
		INNER JOIN RetornoRecomendacionHistorico RHIST WITH(NOLOCK) ON RHIST.marea_id=MA.marea_id
		WHERE CONVERT(VARCHAR(6),HM.production_date,112)>=202111
		AND MA.acodera_chata IS NULL
		--AND RHIST.last_modification<ISNULL(MA.discharge_start_date,HM.discharge_start_date)
		GROUP BY MA.marea_id
		UNION ALL
		--MAREAS SIN RECOMENDACION
		SELECT MA.marea_id,NULL FechaRecomendacion,'Sin recomendación' Clasificacion
		FROM #mareas_acodere MA
		INNER JOIN #historia_mareas HM ON HM.marea_id=MA.marea_id
		LEFT JOIN RetornoRecomendacionHistorico RHIST WITH(NOLOCK) ON RHIST.marea_id=MA.marea_id
		WHERE RHIST.last_modification IS NULL
		AND CONVERT(VARCHAR(6),HM.production_date,112)>=202111
		)MAREAS)GRUPO
WHERE FechaRecomendacion IS NULL OR FechaFinalRec=FechaRecomendacion

--ULTIMA RECOMENDACION DE MAREAS
SELECT DISTINCT REC.*
INTO #ULTREC
FROM #MAREASCLASIF MC
INNER JOIN(
	--MAREAS CERRADAS
	SELECT marea_id,planta_retorno,orden_descarga,chata_descarga,linea_descarga,
	poza_descarga_1,poza_descarga_2,poza_descarga_3,poza_descarga_4
	FROM SPMareasJoinedRecomsHistorico CERRADA WITH(NOLOCK)
	WHERE CONVERT(VARCHAR(6),production_date,112)>=202111
	UNION ALL
	--MAREAS ACTIVAS
	SELECT marea_id,planta_retorno,orden_descarga,chata_descarga,linea_descarga,
	poza_descarga_1,poza_descarga_2,poza_descarga_3,poza_descarga_4
	FROM SPActiveMareasJoinedRecoms ACTIVA WITH(NOLOCK)
	WHERE CONVERT(VARCHAR(6),recom_last_modification,112)>=202111)REC ON REC.marea_id=MC.marea_id

--SIN CAMBIOS: 1945 /10 min ejecucion
--QUERY PRINCIPAL ADHERENCIA
SELECT MWS.marea_id,
		MWS.discharge_plant_name,
		MWS.production_date as fecha_produccion,
		MWS.boat_name,
		FCSD.first_cala_start_date,
		MWS.discharge_plant_arrival_date,
		MWS.eta,
		MWS.declared_ton,
		MWS.actual_ton_discharge,
		OLL.orden_llegada,
		MA.orden_asignado,
		RRH.orden_descarga_global as orden_recomendado,
		MA.motivo_acodere_orden,
		MA.acodera_chata,
		MA.inicio_succion,
		MA.termino_succion,
		MA.desacodera_chata,
		MA.acodera_inicio_succion,
		MA.fin_succion_desacodere,
		MA.acoderar,
		MA.motivo_acodere_inicio,
		MA.motivo_acodere_fin,
		MA.motivo_acodere,
		MA.chata_descarga_acodere,
		--ResChata.chata_descarga as chata_recomendada,--RRH.chata_descarga as chata_recomendada,
		CASE WHEN RRH.chata_descarga='CHATA EXABA' THEN 'EX-ABA'
		WHEN LEFT(RRH.chata_descarga,5)='CHATA' THEN RIGHT(RRH.chata_descarga,LEN(RRH.chata_descarga)-6) 
		ELSE RRH.chata_descarga
		END as chata_recomendada,
		MA.linea_descarga_acodere,
		RRH.linea_descarga as linea_recomendada,
		MA.poza_1,
		MA.poza_2,
		MA.poza_3,
		MA.poza_4,
		RRH.poza_descarga_1 as poza_recomendada_1,
		RRH.poza_descarga_2 as poza_recomendada_2,
		RRH.poza_descarga_3 as poza_recomendada_3,
		NULL poza_recomendada_4, --PENDIENTE DE ACTUALIZAR MODELO
		--Adherencia Orden

		--Para calcular la adherencia de orden se calcula el orden de llegada del barco en el día (el día inicia a las 8 am y termina 8 am del siguiente). El orden real
		--se calcula con el orden de acodere a chata de las embarcaciones. Puede que una embarcación que recibió orden primero, no acodere primero porque alguna otra
		--llegó primero a su chata por aspectos aleatorios (pericia del conductor por ejemplo). En este caso, se da 10 minutos de tolerancia. Es decir, si la embarcación
		--X recibió orden recomendado 1 pero acoderó 3era, se considerará que cumplió la recomendación siempre y cuando la embarcación la diferencia en minutos de acodere
		--entre ella y la embarcación que acoderó primera no sea mayor a 10 minutos.
		siguio_recom_orden=CASE WHEN ABS(DATEDIFF(minute,MA.acodera_chata,(SELECT TOP 1 acodera_chata 
																			FROM (SELECT acodera_chata, ABS(DATEDIFF(second,acodera_chata,MA.acodera_chata)) as diferencia 
																					FROM #mareas_acodere 
																					left join (SELECT DISTINCT id_planta,id_chata, name
																								FROM Chatas_Lineas) as nombres on nombres.name=#mareas_acodere.chata_descarga_acodere
																					WHERE #mareas_acodere.orden_asignado=RRH.orden_descarga_global AND id_planta=MWS.discharge_plant_name) AUX
																			ORDER BY diferencia ASC)))<=10 THEN 1 ELSE 0 END,
	
		--Adherencia Chata-linea
		--Para la adherencia de chata línea se deben cumplir tanto la chata recomendada como la línea recomendada.
		siguio_recom_chata_linea=CASE WHEN (
									CASE WHEN RRH.chata_descarga='CHATA EXABA' THEN 'EX-ABA'
									WHEN LEFT(RRH.chata_descarga,5)='CHATA' THEN RIGHT(RRH.chata_descarga,LEN(RRH.chata_descarga)-6) 
									ELSE RRH.chata_descarga
									END)=MA.chata_descarga_acodere
								AND MA.linea_descarga_acodere=RRH.linea_descarga THEN 1 ELSE 0 END,

		--Adherencia Poza
		--El % de adherencia de pozas es la proporción entre el número de pozas que se recomendó y se usaron, y el número de pozas que se usaron.
		siguio_recom_poza=cast(CASE WHEN 
							--Condicion de haber descargado en al menos una poza
							((CASE WHEN MA.poza_1 IS NOT NULL THEN 1 ELSE 0 END)
								+(CASE WHEN MA.poza_2 IS NOT NULL THEN 1 ELSE 0 END)
								+(CASE WHEN MA.poza_3 IS NOT NULL THEN 1 ELSE 0 END))>0 THEN
							--Siguió Poza 1
							((CASE WHEN (MA.poza_1 IS NOT NULL) AND 
								(RRH.poza_descarga_1=MA.poza_1 OR RRH.poza_descarga_2=MA.poza_1 OR RRH.poza_descarga_3=MA.poza_1) THEN 1 ELSE 0 END)+
							(CASE WHEN (MA.poza_2 IS NOT NULL) AND 
								(RRH.poza_descarga_1=MA.poza_2 OR RRH.poza_descarga_2=MA.poza_2 OR RRH.poza_descarga_3=MA.poza_2) THEN 1 ELSE 0 END)+
							(CASE WHEN (MA.poza_3 IS NOT NULL) AND 
								(RRH.poza_descarga_1=MA.poza_3 OR RRH.poza_descarga_2=MA.poza_3 OR RRH.poza_descarga_3=MA.poza_3) THEN 1 ELSE 0 END))*1.0/
							--Numero de pozas en las que descargó
							((CASE WHEN MA.poza_1 IS NOT NULL THEN 1 ELSE 0 END)
								+(CASE WHEN MA.poza_2 IS NOT NULL THEN 1 ELSE 0 END)
								+(CASE WHEN MA.poza_3 IS NOT NULL THEN 1 ELSE 0 END)) ELSE NULL END as float),

		MA.motivo_linea_descarga,
		MA.motivo_poza,
		RRH.last_modification as hora_recomendacion
		--MC.Clasificacion
INTO #discharge_line_adherence
FROM #historia_mareas MWS
--Este join es para linkear con todo el universo de mareas (con errores de recomendacion y fecha acodere)
inner join #MAREASCLASIF MC ON MC.marea_id=MWS.marea_id 
--Este left join permite obtener la variable first_cala_start_date
left join (SELECT marea_id, MIN(catch_start_date) as first_cala_start_date
			FROM CalasWebService WITH(NOLOCK)
			WHERE catch_amount_ton>0
			GROUP BY marea_id) as FCSD ON FCSD.marea_id=MWS.marea_id
--Este left join incluye el orden de llegada por día (a partir de las 8 am) de cada embarcacion
left join (SELECT marea_id,
		dia_inicio=CASE WHEN DATEPART(hour,discharge_plant_arrival_date)>=13 THEN 
						DATETIMEFROMPARTS(YEAR(discharge_plant_arrival_date),MONTH(discharge_plant_arrival_date),DAY(discharge_plant_arrival_date),13,0,0,0) ELSE
						DATETIMEFROMPARTS(YEAR(DATEADD(day,-1,discharge_plant_arrival_date)),MONTH(DATEADD(day,-1,discharge_plant_arrival_date)),DAY(DATEADD(day,-1,discharge_plant_arrival_date)),13,0,0,0)
					END,
		ROW_NUMBER() OVER (PARTITION BY CASE WHEN DATEPART(hour,discharge_plant_arrival_date)>=13 THEN 
						DATETIMEFROMPARTS(YEAR(discharge_plant_arrival_date),MONTH(discharge_plant_arrival_date),DAY(discharge_plant_arrival_date),13,0,0,0) ELSE
						DATETIMEFROMPARTS(YEAR(DATEADD(day,-1,discharge_plant_arrival_date)),MONTH(DATEADD(day,-1,discharge_plant_arrival_date)),DAY(DATEADD(day,-1,discharge_plant_arrival_date)),13,0,0,0)
					END, discharge_plant_name ORDER BY  discharge_plant_arrival_date) as orden_llegada
			FROM #historia_mareas) as OLL on OLL.marea_id=MWS.marea_id
--Este left join permite tener las chatas por su nombre
left join (SELECT DISTINCT id_chata, name
			FROM Chatas_Lineas) as CHN on CHN.id_chata=MWS.discharge_chata_name
--El siguiente left join permite tener la información de acodere de las embarcaciones
left join #mareas_acodere MA WITH(NOLOCK) ON MA.marea_id=MWS.marea_id
--Luego se agrega SPMareasJoineedRecomsHistorico para obtener el orden
left join SPMareasJoinedRecomsHistorico SPM WITH(NOLOCK)ON SPM.marea_id=MWS.marea_id
--El siguiente left join permite tener la información de recomendaciones
left join RetornoRecomendacionHistorico RRH WITH(NOLOCK) ON RRH.marea_id=MWS.marea_id 
	AND RRH.last_modification=MC.FechaFinalRec
WHERE (RRH.last_modification=MC.FechaFinalRec OR MC.FechaFinalRec IS NULL)


--ACTUALIZAR LINEAS
UPDATE ADH SET
linea_descarga_acodere=NULL
FROM #discharge_line_adherence ADH
WHERE linea_descarga_acodere NOT IN('N','S','0')

--ACTUALIZAR CHATA EXABA EN BD FINAL
UPDATE ADH SET
chata_descarga_acodere='EX-ABA'
FROM #discharge_line_adherence ADH
WHERE chata_descarga_acodere='EX'

--ACTUALIZAR CHATA EXABA EN ULTIMA RECOMENDACION
UPDATE ADH SET
chata_descarga='EX-ABA'
FROM #ULTREC ADH
WHERE chata_descarga='EX'

--QUITAR EL PREFIJO CHATA (DATA GENERADA EN LOS PRIMEROS DIAS DE LA TEMPORADA) EN BD FINAL
UPDATE ADH SET
chata_recomendada=RIGHT(chata_recomendada,LEN(chata_recomendada)-6)
FROM #discharge_line_adherence ADH
WHERE LEFT(chata_recomendada,5)='CHATA'

--QUITAR EL PREFIJO CHATA (DATA GENERADA EN LOS PRIMEROS DIAS DE LA TEMPORADA) EN ULTIMA RECOMENDACION
UPDATE ADH SET
chata_descarga=RIGHT(chata_descarga,LEN(chata_descarga)-6)
FROM #ULTREC ADH
WHERE LEFT(chata_descarga,5)='CHATA'

--ARREGLAR LOS EXABA EN CHATAS RECOMENDADAS EN BD FINAL
UPDATE ADH SET
chata_recomendada='EX-ABA'
FROM #discharge_line_adherence ADH
WHERE chata_recomendada='EXABA'

--ARREGLAR LOS EXABA EN CHATAS RECOMENDADAS EN ULTIMA RECOMENDACION
UPDATE ADH SET
chata_descarga='EX-ABA'
FROM #ULTREC ADH
WHERE chata_descarga='EXABA'


--2768
--QUITAR LOS CASOS QUE NO DEBERIAN ENTRAR EN LA ADHERENCIA (SIN FECHA ACODERE Y SIN RECOMENDACION)
--AGREGAR LOS VALORES DE LA ULTIMA RECOMENDACION (LOS QUE SE VEN EN LA INTERFAZ)
SELECT DS.*,REC.orden_descarga Ultimo_orden,REC.chata_descarga Ultima_chata,REC.linea_descarga Ultima_linea,
REC.poza_descarga_1 Ultima_poza1,REC.poza_descarga_2 Ultima_poza2,REC.poza_descarga_3 Ultima_poza3,
REC.poza_descarga_4 Ultima_poza4
FROM #discharge_line_adherence DS
LEFT JOIN #ULTREC REC ON REC.marea_id=DS.marea_id
WHERE
--FILTRAR CASOS SIN NIGUNA RECOMENDACION
(orden_recomendado IS NOT NULL OR chata_recomendada IS NOT NULL OR linea_recomendada IS NOT NULL
OR poza_recomendada_1 IS NOT NULL OR poza_recomendada_2 IS NOT NULL OR poza_recomendada_3 IS NOT NULL)
--FILTRAR CASOS SIN ACODERE CHATA
AND acodera_chata IS NOT NULL
ORDER BY fecha_produccion,discharge_plant_name,acodera_chata


--ADHERENCIA 
SELECT ROUND(AVG(CAST(siguio_recom_orden as float))*100,2) as perc_adherencia_recom_orden,
		ROUND(AVG(CAST(siguio_recom_chata_linea as float))*100,2) as perc_adherencia_recom_chata_linea,
		ROUND(AVG(CAST(siguio_recom_poza as float))*100,2) as perc_adherencia_recom_poza
FROM #discharge_line_adherence
WHERE 
--FILTRAR CASOS SIN NIGUNA RECOMENDACION
(orden_recomendado IS NOT NULL OR chata_recomendada IS NOT NULL OR linea_recomendada IS NOT NULL
OR poza_recomendada_1 IS NOT NULL OR poza_recomendada_2 IS NOT NULL OR poza_recomendada_3 IS NOT NULL)
--FILTRAR CASOS SIN ACODERE CHATA
AND acodera_chata IS NOT NULL



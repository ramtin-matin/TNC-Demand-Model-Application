import numpy as np
import pandas as pd


def ky_apply_trip_generation(tod1_file, tod2_file, tod3_file, tod4_file, tod5_file, weekdays_csv):
    
    ##read in the files
	overnight = pd.read_csv(tod1_file, index_col = 0)
	morning = pd.read_csv(tod2_file, index_col = 0)
	midday = pd.read_csv(tod3_file, index_col = 0)
	evening = pd.read_csv(tod4_file, index_col = 0)
	night = pd.read_csv(tod5_file, index_col = 0)
    
    
    ##will apply kentucky data, is currently using chicago data
    
    ##apply linear models
	overnight['LINEAR_PICKUPS'] = 0.1251485*overnight['FOOD_EMP'] + 260.4035*overnight['AIR_F'] + 0.0055992*overnight['LOW_INC_0'] + 0.1456829*overnight['HI_INC_0'] + 0.0206288*overnight['LOW_INC_1P'] 
	overnight['LINEAR_PICKUPS_LOG'] =  np.where(overnight['LINEAR_PICKUPS'] == 0, 0, np.log(overnight['LINEAR_PICKUPS']))

	morning['LINEAR_PICKUPS'] = 0.0000952*morning['OTHER_EMP'] + 0.0473473*morning['FOOD_EMP'] + 0.015827*morning['RETAIL_EMP'] + 130.2076*morning['AIR_F'] + 53.18332*morning['TOR_F'] + 0.3745288*morning['HI_INC_0'] + 0.0174729*morning['LOW_INC_0'] + 0.0176063*morning['LOW_INC_1P'] + 0.0000136*morning['HI_INC_1P'] 
	morning['LINEAR_PICKUPS_LOG'] =  np.where(morning['LINEAR_PICKUPS'] == 0, 0, np.log(morning['LINEAR_PICKUPS']))

	midday['LINEAR_PICKUPS'] = 0.1485942*midday['FOOD_EMP'] + 0.0653171*midday['RETAIL_EMP'] + 0.0075034*midday['OTHER_EMP'] + 921.3699*midday['AIR_F'] + 173.4502*midday['TOR_F'] + 0.6336939*midday['HI_INC_0'] + 0.0275177*midday['LOW_INC_0'] + 0.0235304*midday['LOW_INC_1P'] + 0.0000243*midday['HI_INC_1P'] 
	midday['LINEAR_PICKUPS_LOG'] =  np.where(midday['LINEAR_PICKUPS'] == 0, 0, np.log(midday['LINEAR_PICKUPS']))

	evening['LINEAR_PICKUPS'] = 0.1094589*evening['FOOD_EMP'] + 0.0385694*evening['RETAIL_EMP'] + 0.0079508*evening['OTHER_EMP'] + 213.2107*evening['AIR_F'] + 102.1033*evening['TOR_F'] + 0.4440004*evening['HI_INC_0'] + 0.0060732*evening['LOW_INC_0'] + 0.0072242*evening['LOW_INC_1P'] + 0.0000164*evening['HI_INC_1P'] 
	evening['LINEAR_PICKUPS_LOG'] = np.where(evening['LINEAR_PICKUPS'] == 0, 0, np.log(evening['LINEAR_PICKUPS']))

	night['LINEAR_PICKUPS'] = 0.002725*night['OTHER_EMP'] + 0.1846295*night['FOOD_EMP'] + 476.2219*night['AIR_F'] + 0.350088*night['HI_INC_0'] + 0.00000446*night['HI_INC_1P'] + 0.0070386*night['LOW_INC_0'] + 0.0076753*night['LOW_INC_1P']
	night['LINEAR_PICKUPS_LOG'] =  np.where(night['LINEAR_PICKUPS'] == 0, 0, np.log(night['LINEAR_PICKUPS']))
    
    
    
    ##apply negative binomial models
	overnight['PRED_AVG_WD_PICKUPS'] = np.exp(-1.699553*overnight['AIR_F'] + 0.5047479*overnight['TOR_F'] + 0.3486627*overnight['LINEAR_PICKUPS_LOG'] + 1.082214*overnight['LOGSUM'] - 0.0173114*overnight['MEDIAN_AGE'] + 0.0170123*overnight['P_BACH_25P'] + 0.000000716*overnight['TOTAL_EMP_DEN'] - 0.028017*overnight['DEC_18'] + 0.0220474*overnight['JAN_19'] + 0.0560024*overnight['FEB_19'] + 0.1222571*overnight['MAR_19'] + 0.0544462*overnight['APR_19'] + 0.0958898*overnight['MAY_19'] + 0.2011476*overnight['JUN_19'] + 0.217121*overnight['JUL_19'] + 0.2087327*overnight['AUG_19'] + 0.1676383*overnight['SEP_19'] + 0.1657967*overnight['OCT_19'] + 0.2471665*overnight['NOV_19'] + 0.2644915*overnight['DEC_19'] + 0.2964904*overnight['JAN_20'] + 0.2991526*overnight['FEB_20'] - 5.691818)

	morning['PRED_AVG_WD_PICKUPS'] = np.exp(-0.7743786*morning['AIR_F'] + 0.4779832*morning['TOR_F'] + 0.4050372*morning['LINEAR_PICKUPS_LOG'] + 0.6230226*morning['LOGSUM'] - 0.0262*morning['MEDIAN_AGE'] + 0.0143772*morning['P_BACH_25P']+ 0.000000498*overnight['TOTAL_EMP_DEN'] - 0.0563335*morning['DEC_18'] + 0.0918763*morning['JAN_19'] + 0.2369338*morning['FEB_19'] + 0.2762176*morning['MAR_19'] + 0.1257616*morning['APR_19'] + 0.100636*morning['MAY_19'] - 0.0465612*morning['JUN_19'] - 0.0191302*morning['JUL_19'] + 0.0155422*morning['AUG_19'] + 0.1303734*morning['SEP_19'] + 0.1265867*morning['OCT_19'] + 0.1615657*morning['NOV_19'] + 0.101676*morning['DEC_19'] + 0.2673337*morning['JAN_20'] + 0.3775664*morning['FEB_20'] - 1.946639)

	midday['PRED_AVG_WD_PICKUPS'] = np.exp(-0.6405087*midday['AIR_F'] + 0.7809523*midday['TOR_F'] + 0.3927277*midday['LINEAR_PICKUPS_LOG'] + 0.6977969*midday['LOGSUM'] - 0.0208753*midday['MEDIAN_AGE'] + 0.0123951*midday['P_BACH_25P']+ 0.00000161*overnight['TOTAL_EMP_DEN'] - 0.0390817*midday['DEC_18'] + 0.098963*midday['JAN_19'] + 0.2032421*midday['FEB_19'] + 0.2396192*midday['MAR_19'] + 0.1364222*midday['APR_19'] + 0.1078783*midday['MAY_19'] + 0.146028*midday['JUN_19'] + 0.1585698*midday['JUL_19'] + 0.1623656*midday['AUG_19'] + 0.1184939*midday['SEP_19'] + 0.1605051*midday['OCT_19'] + 0.2076105*midday['NOV_19'] + 0.189735*midday['DEC_19'] + 0.2911973*midday['JAN_20'] + 0.3918755*midday['FEB_20'] - 2.551229)

	evening['PRED_AVG_WD_PICKUPS'] = np.exp(-0.4855696*evening['AIR_F'] + 0.890186*evening['TOR_F'] + 0.4090155*evening['LINEAR_PICKUPS_LOG'] + 0.8352944*evening['LOGSUM'] - 0.0216488*evening['MEDIAN_AGE'] + 0.0149257*evening['P_BACH_25P']+ 0.00000217*overnight['TOTAL_EMP_DEN'] - 0.0307113*evening['DEC_18'] + 0.0641547*evening['JAN_19'] + 0.1974895*evening['FEB_19'] + 0.2202914*evening['MAR_19'] + 0.0553759*evening['APR_19'] + 0.0594408*evening['MAY_19'] + 0.0540622*evening['JUN_19'] + 0.0249321*evening['JUL_19'] + 0.0376746*evening['AUG_19'] + 0.056278*evening['SEP_19'] + 0.0935998*evening['OCT_19'] + 0.1979503*evening['NOV_19'] + 0.1853279*evening['DEC_19'] + 0.2399905*evening['JAN_20'] + 0.363544*evening['FEB_20'] - 4.089885)

	night['PRED_AVG_WD_PICKUPS'] = np.exp(0.0188928*night['AIR_F'] + 0.830838*night['TOR_F'] + 0.3445732*night['LINEAR_PICKUPS_LOG'] + 0.9817033*night['LOGSUM'] - 0.0215845*night['MEDIAN_AGE'] + 0.0199614*evening['P_BACH_25P']+ 0.00000224*overnight['TOTAL_EMP_DEN'] - 0.021861*night['DEC_18'] + 0.0234194*night['JAN_19'] + 0.1784418*night['FEB_19'] + 0.2219709*night['MAR_19'] + 0.0799116*night['APR_19'] + 0.0603504*night['MAY_19'] + 0.1119321*night['JUN_19'] + 0.097184*night['JUL_19'] + 0.1199707*night['AUG_19'] + 0.1197228*night['SEP_19'] + 0.148494*night['OCT_19'] + 0.2136709*night['NOV_19'] + 0.256858*night['DEC_19'] + 0.2785087*night['JAN_20'] + 0.4076045*night['FEB_20'] - 5.265743)



    #append all of the tod specific dataframes together
	pred_trips = overnight.append(morning)
	pred_trips = overnight.append(midday)
	pred_trips = overnight.append(evening)
	pred_trips = overnight.append(night)

	#convert average weekday trips to monthly trip totals
	wd = pd.read_csv(weekdays_csv)
	pred_trips = pred_trips.merge(wd, on = ['MONTH','YEAR'])
	
	return pred_trips
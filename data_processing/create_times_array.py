from datetime import datetime, timedelta
from decorators import benchmark

@benchmark
def create_tdeltas_array(file_list):
    FMT = '%H%M%S'
    tdeltas = [0]
    if file_list[0][-12:-5] == 'updated':
        time_str_ref = file_list[0][-21:-15]
        for fits_file in file_list[1:]:
            time_str = fits_file[-21:-15]
            tdeltas.append( timedelta.total_seconds(datetime.strptime(time_str, FMT) - datetime.strptime(time_str_ref, FMT)) )
        print('tdeltas array calculated')
    else:
        time_str_ref = file_list[0][-16:-10]
        for fits_file in file_list[1:]:
            time_str = fits_file[-16:-10]
            tdeltas.append( timedelta.total_seconds(datetime.strptime(time_str, FMT) - datetime.strptime(time_str_ref, FMT)) )
        print('tdeltas array calculated')
    return tdeltas
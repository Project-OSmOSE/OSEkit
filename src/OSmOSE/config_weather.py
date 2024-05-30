
def logarithmic(x, a, b, offset):
	return 10**(((x-offset) + 10*a*np.log10(8000)) / (20*b))

def quadratic(x, a, b, c, offset):
	return a*(x-offset)**2 + b*(x-offset) + c

empirical = {
	"Hildebrand": {
		"frequency": 8000,
		"samplerate": 200000,
		"preprocessing": {
			"nfft": 2000,
			"window_size": 2000,
			"spectro_duration": 300,
			"window": "hanning",
			"overlap": 0
			},
		"function": logarithmic,
		"averaging_duration": 3600,
		"parameters": {
			"a": 78,
			"b": 1.5
			}
		},

	"Pensieri": {
		"frequency": 8000,
		"samplerate": 100000,
		"preprocessing": {
			"nfft": 1024,
			"window_size": 1024,
			"spectro_duration": 0.8192,
			"window": "hanning",
			"overlap": 0
			},
		"function": quadratic,
		"averaging_duration": 4.5,
		"parameters": {
			"a": 0.044642,
			"b": -3.2917,
			"c": 63.016
			}
		}
}


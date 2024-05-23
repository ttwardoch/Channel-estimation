import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import scipy as sc


def correlate(x, y, mode="same"):
    # Function purely to increase speed by finding only some lags
    # Use np.correlate in channel estimation instead
    x = np.array(x)
    y = np.array(y)
    c_0 = np.sum(x * np.conjugate(y))
    c_1 = np.sum(np.roll(x, -1) * np.conjugate(y))
    c_2 = np.sum(np.roll(x, -2) * np.conjugate(y))
    c__2 = np.sum(np.roll(x, 2) * np.conjugate(y))
    c__1 = np.sum(np.roll(x, 1) * np.conjugate(y))

    return np.array([c__2, c__1, c_0, c_1, c_2])

SCS = 30000
PURE_AWGN = False
ITERATIONS = 10000
ALL_PLOTS = False
ESTIMATION = True
pad_length = 200

z = np.load("model.npz")
amplitudes = np.abs(z["bins"])

freq = z["bins_labels"]
average_freq = (freq[-1] + freq[0])/2
freq -= average_freq # Get rid of Doppler shift

length_of_D = math.ceil((freq[-1] - freq[0])/SCS)
indices = [i-math.floor(length_of_D/2) for i in range(length_of_D)]

D = np.zeros(length_of_D, dtype=np.complex128)

for a, f in zip(z["bins"], freq):
    index = math.floor(f/SCS+0.5) + math.floor(length_of_D/2)
    D[index] += a

D /= D[math.floor(length_of_D/2)] # make index 0 phase 0

D_amplitude = np.abs(D)
D = D/np.sum(D_amplitude)
D_amplitude = np.abs(D_amplitude)/np.sum(D_amplitude)

print(D)

plt.bar(freq, amplitudes, width=freq[1]-freq[0])
if ALL_PLOTS: plt.show()
plt.close()

plt.bar(indices, D_amplitude, width=1)
plt.title("Amplitude of resampled Doppler spectrum")
plt.xlabel("Index of sub-carrier")
plt.ylabel("Normalised amplitude")
if ALL_PLOTS: plt.show()
plt.close()

for modulation in ["BPSK", "QPSK", "16-QAM", "64-QAM"]:
    SNRs = []
    BERs = []
    for log_SNR in tqdm(np.linspace(-4, 6, 1000)):
        SNR = 10**log_SNR
        std_deviation = np.sqrt(2/SNR)
        if modulation == "BPSK":
            input_symbols = np.random.choice([1, -1], ITERATIONS)
        elif modulation == "QPSK":
            input_symbols = np.random.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], ITERATIONS) / np.sqrt(2)
        elif modulation == "16-QAM":
            input_symbols = np.random.choice([3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j, 1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j, -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j, -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j], ITERATIONS) / np.sqrt(10)
        elif modulation == "64-QAM":
            constellation = []
            for a in [-7, -5, -3, -1, 1, 3, 5, 7]:
                for b in [-7, -5, -3, -1, 1, 3, 5, 7]:
                    constellation.append(a + b*1j)
            input_symbols = np.random.choice(constellation, ITERATIONS) / np.sqrt(42)

        # Channel
        # after_convolution = input_symbols
        after_convolution = np.convolve(input_symbols, D, mode="same")
        if PURE_AWGN:
            after_convolution = input_symbols
        noise = np.random.normal(loc=0.0, scale=std_deviation, size=np.size(after_convolution)) + np.random.normal(loc=0.0, scale=std_deviation, size=np.size(after_convolution)) * 1j
        after_addition = after_convolution + noise

        # Channel estimation
        if ESTIMATION:
            pilots = [symbol if i % 10 == 0 else 0 for i, symbol in enumerate(input_symbols)]
            correlation = correlate(after_addition, pilots, mode="same")
            #plt.plot(np.abs(autocorrelation))
            #plt.show()
            zoom = correlation[np.size(correlation)//2-2:np.size(correlation)//2+3]
            #zoom = D
            zoom /= np.sum(np.abs(zoom))
            measured = zoom
            #print(zoom)
            absolute = np.abs(zoom)
            normalised = absolute/np.sum(absolute)
            #plt.plot(normalised)
            #plt.show()

            zoom = np.pad(zoom, (pad_length, pad_length))

            n = np.size(zoom)//2

            zoom = np.roll(zoom, n+1)
            time_domain = sc.fft.ifft(zoom)
            time_domain = 1/time_domain
            #print(time_domain)
            inverse_filter = sc.fft.fft(time_domain)
            inverse_filter = np.roll(inverse_filter, n)

            #print(inverse_filter)

            estimated = np.convolve(after_addition, inverse_filter, mode="same")

            #estimated, r = sc.signal.deconvolve(np.pad(after_addition, (4, 4)), zoom)
            #estimated = estimated[4:]
        else:
            estimated = after_addition

        #print(np.size(estimated))
        #print(estimated)
        #print(input_symbols)

        good = 0
        total = 0
        if modulation == "BPSK":
            for x, y in zip(input_symbols, estimated):
                total += 1
                if np.real(x)*np.real(y) > 0:
                    good += 1
        elif modulation == "QPSK":
            average_power = np.average(np.abs(estimated * estimated))
            scale = np.sqrt(2) / np.sqrt(average_power)
            constellation = estimated * scale
            for x, y in zip(input_symbols, estimated):
                total += 2
                if np.real(x) * np.real(y) > 0:
                    good += 1
                if np.imag(x) * np.imag(y) > 0:
                    good += 1
        elif modulation == "16-QAM":
            average_power = np.average(np.abs(estimated * estimated))
            scale = np.sqrt(10)/np.sqrt(average_power)
            constellation = estimated * scale
            for x, y in zip(input_symbols*np.sqrt(10), constellation):
                total += 4
                if np.real(x) * np.real(y) > 0:
                    good += 1
                if np.imag(x) * np.imag(y) > 0:
                    good += 1
                if (0 < np.real(x) < 2 or np.real(x) < -2) and (0 < np.real(y) < 2 or np.real(y) < -2):
                    good += 1
                if (0 > np.real(x) > -2 or np.real(x) > 2) and (0 > np.real(y) > -2 or np.real(y) > 2):
                    good += 1
                if (0 < np.imag(x) < 2 or np.imag(x) < -2) and (0 < np.imag(y) < 2 or np.imag(y) < -2):
                    good += 1
                if (0 > np.imag(x) > -2 or np.imag(x) > 2) and (0 > np.imag(y) > -2 or np.imag(y) > 2):
                    good += 1
        elif modulation == "64-QAM":
            average_power = np.average(np.abs(estimated * estimated))
            scale = np.sqrt(42)/np.sqrt(average_power)
            constellation = estimated * scale
            for x, y in zip(input_symbols*np.sqrt(42), constellation):
                total += 6
                if np.real(x) * np.real(y) > 0:
                    good += 1

                if np.imag(x) * np.imag(y) > 0:
                    good += 1

                if -4 < np.real(x) < 4 and -4 < np.real(y) < 4:
                    good += 1
                elif (np.real(x) > 4 or np.real(x) < -4) and (np.real(y) > 4 or np.real(y) < -4):
                    good += 1

                if -4 < np.imag(x) < 4 and -4 < np.imag(y) < 4:
                    good += 1
                elif (np.imag(x) > 4 or np.imag(x) < -4) and (np.imag(y) > 4 or np.imag(y) < -4):
                    good += 1

                if (-6 > np.real(x) or -2 < np.real(x) < 2 or 6 < np.real(x)) and (-6 > np.real(y) or -2 < np.real(y) < 2 or 6 < np.real(y)):
                    good += 1
                elif ((2 < np.real(x) < 6) or (-6 < np.real(x) < -2)) and ((2 < np.real(y) < 6) or (-6 < np.real(y) < -2)):
                    good += 1

                if (-6 > np.imag(x) or -2 < np.imag(x) < 2 or 6 < np.imag(x)) and (-6 > np.imag(y) or -2 < np.imag(y) < 2 or 6 < np.imag(y)):
                    good += 1
                elif ((2 < np.imag(x) < 6) or (-6 < np.imag(x) < -2)) and ((2 < np.imag(y) < 6) or (-6 < np.imag(y) < -2)):
                    good += 1
        BER = 1 - good/total
        BERs.append(BER)

        average_power_signal = np.average(np.abs(after_convolution * after_convolution))
        average_power_noise = np.average(np.abs(noise * noise))
        SNR = 10*np.log10(average_power_signal / average_power_noise)

        """
        SNR_BOUND = 10
        if modulation == "QPSK" and SNR_BOUND + 1 > SNR > SNR_BOUND:
            plt.scatter(np.real(constellation[2:-2]), np.imag(constellation[2:-2]))
            plt.title("QPSK constellation: Doppler")
            plt.xlabel("Real value")
            plt.ylabel("Imaginary value")
            print(SNR)
            plt.show()
        """
        SNRs.append(SNR)
    if ESTIMATION and ALL_PLOTS:
        plt.bar(indices, np.abs(measured), width=1)
        plt.title("Amplitude of the Doppler spectrum estimate")
        plt.xlabel("Index of sub-carrier")
        plt.ylabel("Normalised amplitude")
        plt.show()

    plt.plot(SNRs, BERs, label=modulation)

plt.yscale("log")
plt.legend()
plt.title(f"FIR of length {5+pad_length*2}")
#plt.title(f"Doppler + AWGN channel")
plt.xlabel("Signal to noise ratio [dB]")
plt.ylabel("Bit error rate")
plt.xlim(-30, 50)
plt.ylim(0.0001, 1)
plt.grid(visible=True)
plt.show()


"""
NoiseFirst 算法实现
"""
import math
import numpy as np


class NoiseFirst:

    def __init__(self, noiseHist, epsilon):
        """
        :param noiseHist: 加噪后的序列
        :param epsilon: 隐私预算参数
        """
        self.epsilon = epsilon
        self.noise_hist = noiseHist
        self.P = []
        self.PP = []
        self.K = self.N = len(noiseHist)
        self.solution = []
        self.SSEStar = []  #minT(i, k)
        self.resultHist = []

#
    def findOptK(self, hist):
        """
        返回误差和最小的分组数
        :param hist: 加噪后的序列
        :param epsilon:
        :return:
        """
        self.initialPara(hist)
        self.SSEStar = np.zeros((self.K, self.N), dtype=np.float)
        for i in range(0, self.K):
            self.SSEStar[i][i] = 0
        for i in range(1, self.K):
            avg = self.P[i] / (i + 1)
            differ = 0
            for j in range(0, i + 1):
                v = abs(hist[j] - avg)
                differ += v*v
            self.SSEStar[i][0] = differ

        for k in range(self.K):
            for i in range(k + 1, self.N):
                self.innerLoop(k - 1, i, k)

        minValue = float('inf')
        for i in range(self.K):
            estimatedErr = self.SSEStar[self.N-1][i] - (2.0*(self.N-2.0*(i+1)))/(pow(self.epsilon, 2.0))
            if estimatedErr < minValue:
                minValue = estimatedErr
                optK = i+1
        return optK


    def initialPara(self, hist):
        """
        初始化P与PP
        :param hist:
        :return:
        """
        self.P.append(hist[0])
        self.PP.append(math.pow(hist[0], 2))
        for index, elem in enumerate(hist[1:]):
            self.P.append(elem + self.P[index])
            self.PP.append(math.pow(elem, 2) + self.PP[index])

        self.solution = np.zeros((self.K, self.K), dtype=np.int)
        for i in range(0, self.K):
            self.solution[i][i] = i


#
    def getResultHist(self, noiseHist, k):
        partition = self.collParStra(k)
        for i in range(0, len(partition) - 1):
            self.addNoise2Partition(noiseHist, partition[i], partition[i + 1])
        return self.resultHist


    def innerLoop(self, startIndex, i, k):
        minDist = float('inf')
        for j in range(startIndex, i):
            tempSSE = self.SSEStar[j][k-1] + self.calSSE(j + 1, i)
            if tempSSE < minDist:
                self.solution[i][k] = j + 1
                self.SSEStar[i][k] = tempSSE
                minDist = tempSSE


    def addNoise2Partition(self, noiseHist, begin, end):

        if end == begin + 1:
            self.resultHist.append(noiseHist[begin])
            return

        n = end - begin
        avg = 0
        for i in range(begin, end):
            avg += noiseHist[i]
        avg /= n
        HbarDbar_merge = 0
        for i in range(begin, end):
            HbarDbar_merge += math.pow(abs(avg - noiseHist[i]), 2.0)

        mergeEstimateError = HbarDbar_merge - (2.0 * (n - 2)) / (self.epsilon * self.epsilon)
        dworkEstimateError = (2.0 * n) / (self.epsilon * self.epsilon)

        if mergeEstimateError > dworkEstimateError:
            for i in range(begin, end):
                self.resultHist.append(noiseHist[i])
        else:
            for i in range(begin, end):
                self.resultHist.append(avg)
        return self.resultHist


    def calSSE(self, i, j):
        if i == j:
            return 0.0
        SSE_ij = self.PP[j] - self.PP[i - 1] - pow(self.P[j] - self.P[i - 1], 2) / (j - i + 1)
        return SSE_ij


    def collParStra(self, k):
        boundary = [0] * (k+1)
        boundary[k] = self.N
        boundary[0] = 0

        n = self.N-1
        for i in reversed(range(1, k)):
            j = self.solution[n][i]
            if n == i:
                for x in range(1, n + 1):
                    boundary[x] = x
                    break
            boundary[i] = j
            n = j - 1
        return boundary















class EWMAMAAttackDetector:
    def __init__(self, lambda_value=0.05, window=3, L=2):
        self.lambda_value = lambda_value
        self.window = window
        self.L = L
        self.sr_history = deque(maxlen=window)
        self.ewma_value = 0.0
        self.sum_sr = 0.0
        self.esr_value = 0.0
        self.esr_values = []
        self.anomaly_steps = []

    def variance_ma(self, n, step):
        if step < self.window:
            return (n * (n + 1) * (2 * n + 1)) / (6 * step)
        else:
            return (n * (n + 1) * (2 * n + 1)) / (6 * self.window)

    def covariance_ma(self, n, k1, k2):
        if k1 < self.window and k2 < self.window:
            return (n * (n + 1) * (2 * n + 1)) / (6 * k2)
        elif (k2 - k1) < self.window:
            overlap = (k1 - k2 + self.window)
            return overlap * (n * (n + 1) * (2 * n + 1)) / (6 * self.window ** 2)
        else:
            return 0

    def variance_esr(self, n, step):
        var_esr = 0.0
        for k in range(step):
            var_esr += (
                (self.lambda_value ** 2)
                * ((1 - self.lambda_value) ** (2 * k))
                * self.variance_ma(n, step - k)
            )
        for k1 in range(1, step):
            for k2 in range(k1 + 1, step + 1):
                var_esr += (
                    2
                    * (self.lambda_value ** 2)
                    * ((1 - self.lambda_value) ** (2 * step - k1 - k2))
                    * self.covariance_ma(n, k1, k2)
                )
        return var_esr

    def control_limits(self, n, step):
        var_esr = self.variance_esr(n, step)
        if var_esr < 0:
            var_esr = 0.0
        ucl = self.L * np.sqrt(var_esr)
        lcl = -ucl
        return ucl, lcl

    def update(self, actual, predicted, step, label=None):
        sr_value = compute_signed_rank(actual, predicted)
        if len(self.sr_history) == self.window:
            self.sum_sr -= self.sr_history[0]
        self.sr_history.append(sr_value)
        self.sum_sr += sr_value

        ma_value = self.sum_sr / len(self.sr_history)

        self.esr_value = (
            self.lambda_value * ma_value + (1 - self.lambda_value) * self.esr_value
        )
        self.esr_values.append(self.esr_value)

        n = len(self.sr_history)
        ucl, lcl = self.control_limits(n, step)

        self.attack_detected = not (lcl <= self.esr_value <= ucl)
        self.predicted_labels.append(int(self.attack_detected))
        if label is not None:
            self.labels.append(label)
        if self.attack_detected:
            self.anomaly_steps.append(step)
            print(f"Anomaly detected at step {step} with ESR value: {self.esr_value}")
        return self.attack_detected
    def get_esr_values(self):
        return self.esr_value

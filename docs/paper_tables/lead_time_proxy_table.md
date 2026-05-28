# Lead Time Proxy Table

This table reports a sample-level lead-time proxy based on `y_remaining_true` for correctly predicted risk samples. It is not a strict first-alarm lead time because simulation IDs and time indices are not stored in the current prediction files.

| Method                  | Alarm recall   | Alarm precision   | Mean lead time   | Median lead time   | Lead time Q25   | Lead time Q75   |
|:------------------------|:---------------|:------------------|:-----------------|:-------------------|:----------------|:----------------|
| Traditional EWS         | 0.78 ± 0.04    | 0.67 ± 0.06       | 13.81 ± 0.30     | 13.40 ± 0.55       | 6.80 ± 0.45     | 20.40 ± 0.55    |
| Image only              | 0.80 ± 0.04    | 0.68 ± 0.04       | 13.89 ± 0.23     | 13.20 ± 0.45       | 7.00 ± 0.00     | 20.80 ± 0.45    |
| Classic patch           | 0.80 ± 0.05    | 0.71 ± 0.09       | 14.09 ± 0.28     | 13.60 ± 0.55       | 6.80 ± 0.45     | 21.00 ± 0.00    |
| Dynamic patch v1        | 1.00 ± 0.00    | 0.40 ± 0.02       | 15.18 ± 0.13     | 15.00 ± 0.00       | 8.00 ± 0.00     | 22.65 ± 0.49    |
| Classic + dynamic patch | 0.80 ± 0.07    | 0.71 ± 0.05       | 14.14 ± 0.20     | 13.40 ± 0.55       | 7.00 ± 0.00     | 21.20 ± 0.45    |
| Full                    | 0.79 ± 0.05    | 0.73 ± 0.08       | 14.02 ± 0.26     | 13.40 ± 0.55       | 6.80 ± 0.45     | 21.00 ± 0.00    |

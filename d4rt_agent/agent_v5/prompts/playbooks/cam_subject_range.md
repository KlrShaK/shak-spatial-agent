- Call `distance` with a = the subject and b = 'cam'.
- Read the per-span 'reading' column in time order: closer, farther, still. Closer then farther is a reversal.
- A ratio near x1.0 with nothing beyond the threshold means the distance stays about the same.
- 'closer_then_farther' or 'farther_then_closer' is a reversal: options call it fluctuating / back and forth,
  not approaching or retreating.

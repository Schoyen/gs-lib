pub fn double_factorial(n: i64) -> f64 {
    if n < -1 {
        return 0.0;
    }

    if n <= 0 {
        return 1.0;
    }

    let mut val = 1.0;
    let mut k = n;

    while k > 0 {
        val *= k as f64;
        k -= 2;
    }

    val
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_double_factorial() {
        assert_abs_diff_eq!(double_factorial(-2), 0.0);
        assert_abs_diff_eq!(double_factorial(0), 1.0);
        assert_abs_diff_eq!(double_factorial(1), 1.0);
        assert_abs_diff_eq!(double_factorial(2), 2.0);
        assert_abs_diff_eq!(double_factorial(3), 3.0);
        assert_abs_diff_eq!(double_factorial(4), 8.0);
        assert_abs_diff_eq!(double_factorial(5), 15.0);
        assert_abs_diff_eq!(double_factorial(10), 3840.0);
    }
}

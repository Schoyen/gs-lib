pub fn double_factorial(n: i64) -> u64 {
    if n < -1 {
        return 0;
    }

    if n <= 0 {
        return 1;
    }

    let mut val: u64 = 1;
    let mut k = n;

    while k > 0 {
        val *= k as u64;
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
        assert_abs_diff_eq!(double_factorial(-2), 0);
        assert_abs_diff_eq!(double_factorial(0), 1);
        assert_abs_diff_eq!(double_factorial(1), 1);
        assert_abs_diff_eq!(double_factorial(2), 2);
        assert_abs_diff_eq!(double_factorial(3), 3);
        assert_abs_diff_eq!(double_factorial(4), 8);
        assert_abs_diff_eq!(double_factorial(5), 15);
        assert_abs_diff_eq!(double_factorial(10), 3840);
    }
}

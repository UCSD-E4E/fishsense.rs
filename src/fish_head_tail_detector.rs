use ndarray::{arr1, array, s, stack, Array1, Array2, Array3, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};

pub struct FishHeadTailDetector {

}

// trait Nonzero<T> {

// }

// impl<T> Nonzero<T> for Vec<(i32, i32)> {

// }

impl FishHeadTailDetector {
    fn find_head_tail(mask: Array2<u8>) {
        let nonzero: Vec<(i32, i32)> = mask
        .indexed_iter()
        .filter_map(|(index, &item)| if item != 0 {Some((index.0 as i32, index.1 as i32))} else { None })
        .collect();        
        
        // Note: Fix everything so that its floats instead of unsigned integers
        let (y, x): (Vec<_>, Vec<_>) = nonzero.iter().cloned().unzip();

        let x_min = *x.iter().min().unwrap() as usize;
        let y_min = *y.iter().min().unwrap() as usize;
        let x_max = *x.iter().max().unwrap() as usize;
        let y_max = *x.iter().max().unwrap() as usize;
        

        let x_mean = (x.iter().sum::<i32>() as f64)/(x.len() as f64);
        let y_mean = (y.iter().sum::<i32>() as f64)/(y.len() as f64);

        let mask_crop = mask.slice(s![y_min..y_max,x_min..x_max]);

        let mut new_x = vec![0 as f64; x.len()];
        let mut new_y = vec![0 as f64; y.len()];

        for i in 0..x.len() {
            new_x[i] = x[i] as f64 - x_mean;
            new_y[i] = y[i] as f64 - y_mean;
        }

        let x_min = new_x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_min = new_y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = new_x.iter().fold(f64::INFINITY, |a, &b| a.max(b));
        let y_max = new_y.iter().fold(f64::INFINITY, |a, &b| a.max(b));
        
        println!("{:?}", nonzero);
        println!("{:?}", y);
        println!("{:?}", x);
        println!("{:?}", mask_crop);
        
        let coords = vec![new_x, new_y];
        // let correlation = 

    } 
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mask: Array2<u8> = array![
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ];
        FishHeadTailDetector::find_head_tail(mask);
        assert_eq!(4, 4);
    }
}

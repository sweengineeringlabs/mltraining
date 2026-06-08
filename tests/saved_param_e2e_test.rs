// @covers: SavedParam
use mltraining::SavedParam;

#[test]
fn test_saved_param_stores_data_and_shape() {
    let p = SavedParam {
        data: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![2, 2],
    };
    assert_eq!(p.data.len(), 4);
    assert_eq!(p.shape, vec![2, 2]);
}

#[test]
fn test_saved_param_data_values_are_preserved() {
    let data = vec![0.1_f32, 0.2, 0.3];
    let p = SavedParam { data: data.clone(), shape: vec![3] };
    for (a, b) in p.data.iter().zip(data.iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}

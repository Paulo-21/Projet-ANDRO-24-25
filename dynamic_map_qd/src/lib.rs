use pyo3::{prelude::*, types::{PyDict, PyList}};
use rand::prelude::*;
use std::{collections::HashMap, env, fs::File, io::Write};
use litemap::LiteMap;

fn cmp_prec_is_same(a : &[f64], b : &[f64], epsilon : &[f64]) -> bool {
    for ((e1,e2), eps) in a.iter().zip(b).zip(epsilon) {
        if (e1-e2).abs() >= *eps { return false; }
    }
    true
}

#[derive(Clone, IntoPyObject, Debug)]
struct Individual {
    fitness: u32,
    coef_mutation: f64,
    genotype : Vec<f64>,
}
impl Individual {
    fn new(fitness: u32, genotype : Vec<f64>) -> Self {
        Self { fitness, coef_mutation: 0.01, genotype }
    }
}

#[derive(Debug, Clone, Default)]
enum DynaNode {
    #[default]
    Empty,
    Leaf{
        point_index : usize,
        depth : u32,
        lb : Vec<f64>,
        ub : Vec<f64>,
    },
    Node {
        submap : LiteMap<usize, DynaNode>,
        depth : u32,
        lb : Vec<f64>,
        ub : Vec<f64>,
        center : Vec<f64>
    }
}
impl DynaNode {
    fn insert_or_divide(&mut self, genome : Vec<f64>, fitness_new : u32, behavior_descriptor : Vec<f64>, mut refv : &mut Vec<Individual>, max_precision : &[f64]) -> bool {
    //fn sub_divide(&mut self, genome : Vec<f64>, fitness_new : u32, behavior_descriptor : Vec<f64>, mut refv : &mut Vec<Individual>, max_precision : &[f64]) -> bool {
        match std::mem::take(self) {
            DynaNode::Leaf { point_index, depth, lb, ub } => {
                println!("ALREADY LEAF");
                if depth >= 3 || cmp_prec_is_same(&refv[point_index].genotype, &genome, &max_precision) {
                    if fitness_new <= refv[point_index].fitness {
                        println!("REJECTED BRO");
                        return false;
                    }
                    refv[point_index].genotype = genome;
                    refv[point_index].fitness = fitness_new;
                    return true;
                }
                let center = DynaNode::compute_center(&lb, &ub);
                let (idx, nub, nlb) = DynaNode::compute_index_and_lub(&genome, &center, &ub, &lb);
                let (idx2, nub2, nlb2) = DynaNode::compute_index_and_lub(&refv[point_index].genotype, &center, &ub, &lb);
                let mut submap = LiteMap::new();
                submap.insert(idx, DynaNode::Leaf {
                    point_index,
                    depth: depth + 1,
                    ub: nub,
                    lb: nlb,
                });
                match submap.get_mut(&idx2) {
                    Some(x) => {
                        x.insert_or_divide(genome, fitness_new, behavior_descriptor, refv, max_precision);
                    },
                    None => {
                        refv.push(Individual { fitness: fitness_new, coef_mutation: 0.01, genotype:genome });
                        submap.insert(idx2, DynaNode::Leaf {
                            point_index : refv.len()-1,
                            depth: depth + 1,
                            ub: nub2,
                            lb: nlb2,
                        });
                    }
                }
                //Create new node
                *self = DynaNode::Node { submap, depth, lb, ub, center };
            }
            DynaNode::Node { mut submap, depth, lb, ub, center } => {
                let (idx, newub, nwlb) = DynaNode::compute_index_and_lub(&genome, &center, &ub, &lb);
                println!("lower : {:?}", lb);
                println!("upper : {:?}", ub);
                match submap.get_mut(&idx) {
                    Some(x) => {
                        (*x).insert_or_divide(genome, fitness_new, behavior_descriptor, &mut refv, max_precision);
                    },
                    None => {
                        println!("Inserted in an another subspace");
                        refv.push(Individual { fitness: fitness_new, coef_mutation: 0.01, genotype: genome });
                        submap.insert(idx, DynaNode::Leaf { point_index: refv.len()-1, depth: depth+1, lb: nwlb, ub: newub });
                    }
                }
                *self = DynaNode::Node { submap, depth, lb, ub, center };  
            },
            _ => {}
        }
        true
    }
    fn compute_center(a : &[f64], b : &[f64]) -> Vec<f64> {
        //return a.iter().zip(b).map(|(a1, b1)| a1+b1/2.).collect();
        let mut c = Vec::with_capacity(a.len());
        for (a1, b1) in a.iter().zip(b) { c.push((a1 + b1)/2.0); }
        c
    }
    #[allow(dead_code)]
    fn compute_index(a : &[f64], center : &[f64]) -> usize {
        let mut index = 0;
        for (k,(a, c)) in a.iter().zip(center).enumerate() {
            if a >= c { index |= 1<<k; }
        }
        index
    }
    fn compute_index_and_lub(a : &[f64], center : &[f64], old_up : &[f64], old_lb : &[f64]) -> (usize, Vec<f64>, Vec<f64>) {
        let mut index = 0;
        let mut ub: Vec<f64> = Vec::with_capacity(a.len());
        let mut lb: Vec<f64> = Vec::with_capacity(a.len());
        for (k,(a, c)) in a.iter().zip(center).enumerate() {
            if a >= c {
                index |= 1<<k;
                ub.push(old_up[k]);
                lb.push(*c);
            }
            else {
                ub.push((*c).next_down());
                lb.push(old_lb[k]);
            }
        }
        (index, ub, lb)
    }
}
struct DynaMap {
    saved_point: Vec<Individual>,
    root: HashMap<Vec<u64>, DynaNode>,
    max_depth : u32,
    max_precision : Vec<f64>,
}
impl DynaMap {
    pub fn new(max_depth : Option<u32>, max_precision : Option<Vec<f64>>) -> Self {
        Self {
            saved_point: Vec::new(),
            root: HashMap::new(),
            max_depth : max_depth.unwrap_or(3),
            max_precision : max_precision.unwrap_or(vec![0.001;6]),
        }
    }
    fn insert(&mut self, behavior_descriptor : Vec<f64>, genome : Vec<f64>, fitness_new : u32) -> bool {
        if genome.is_empty() { return false; }
        let mut conv = Vec::with_capacity(genome.capacity());
        conv.extend(behavior_descriptor.iter().map(|&a| a.to_bits()));
        println!("TRY  {:?}", genome);
        if let Some(p) = self.root.get_mut(&conv) {
            println!("Behavior already in hashmap");
            p.insert_or_divide(genome, fitness_new, behavior_descriptor, &mut self.saved_point, &self.max_precision);
        }
        else {
            println!("{:?}", genome);
            let indiv = Individual::new(fitness_new, genome);
            self.saved_point.push(indiv);
            println!("{:?}",behavior_descriptor);
            self.root.insert(conv,
                DynaNode::Leaf {
                    point_index : self.saved_point.len()-1, depth : 0,
                    ub : behavior_descriptor.iter().map(|el| {
                        let s = el.to_string();
                        let spl :Vec<&str> = s.split('.').collect();
                        let prec = spl[1].len() as i32;
                        (el + 10f64.powi(-prec)).next_down()
                    }).collect(),
                    lb : behavior_descriptor
                });
            println!("INSERT NEW IN HASHMAP");
        }
        
        true
    }
}
#[pyclass]
pub struct Archive {
    archive_map: DynaMap,
    #[pyo3(get, set)]
    dynamic_application: String,
    #[pyo3(get, set)]
    csv_archive_name: String,
    //#[pyo3(get, set)]
    //stock_path: String,
}
#[pymethods]
impl Archive {
    #[new]
    #[pyo3(signature = (dynamic_application/*, stock_path*/, max_depth=3, max_precision=vec![0.001;3]))]  // Définit les arguments optionnels
    pub fn new(dynamic_application: String, /*stock_path: String,*/ max_depth : Option<u32>, max_precision : Option<Vec<f64>>) -> Self {
        //let mut archive_name = stock_path.clone();
        let mut csv_archive_name = String::new();
        csv_archive_name.push_str("/test.csv");

        Self {
            archive_map: DynaMap::new( max_depth, max_precision),
            dynamic_application,
            //stock_path,
            csv_archive_name,
        }
    }
    fn create_csv_archive_file(&mut self, dynamic_application: String, simulator_scene_simualtion:Bound<'_, PyAny>, simulator:String) {
        let object_to_grasp: String = simulator_scene_simualtion
            .getattr("object_to_grasp").unwrap()
            .extract().unwrap();
        let gripper: String = simulator_scene_simualtion
            .getattr("gripper").unwrap()
            .extract().unwrap();

        // Construction du chemin
        let base_path = env::current_dir().unwrap().join("result_archive");
        let _ = std::fs::create_dir_all(&base_path); // crée le dossier si nécessaire

        let mut version = 1;
        let mut csv_archive_name = base_path.join(format!(
            "{}_object_{}_robot_{}_{}.csv",
            simulator, object_to_grasp, gripper, dynamic_application
        ));

        // Incrémente jusqu'à trouver un nom de fichier libre
        while csv_archive_name.exists() {
            version += 1;
            csv_archive_name = base_path.join(format!(
                "{}_object_{}_robot_{}_{}_v{}.csv",
                simulator, object_to_grasp, gripper, dynamic_application, version
            ));
        }

        // Enregistre le nom et crée le fichier
        self.csv_archive_name = csv_archive_name.to_string_lossy().to_string();
        let _ = File::create(&self.csv_archive_name);

    }
    /// Formats the sum of two numbers as string.
    fn store_one_new_element_in_archive(&mut self, a: Bound<'_, PyDict>) -> PyResult<bool> {
        let behavior = if let Ok(Some(behavior)) = a.get_item("behavior_descriptor") {
            behavior.extract().unwrap()
        }else { Vec::new() };
        let fitness_new: u32 = if let Ok(Some(fitness)) = a.get_item("fitness") {
             fitness.extract::<u32>()?
        } else { 0 };
        let genome = if let Ok(Some(genome)) = a.get_item("genome") {
            genome.extract().unwrap()
        }else {
            Vec::new()
        };
        let inserted = self.archive_map.insert(behavior, genome, fitness_new);
        Ok(inserted)
    }
    #[allow(non_snake_case)]
    fn select_best_individuals_from_archive(&self, size : usize) -> PyResult<Vec<Vec<f64>>> {
        let mut ord_vec = self.archive_map.saved_point.clone();
        ord_vec.sort_unstable_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        let best = Vec::with_capacity(size);
        Ok(best)
    }
    /*fn order_archive_by_fitness(&self) -> Vec<DynaNode> {
        let mut ord_vec = self.archive_map.saved_point.clone();
        ord_vec.sort_unstable_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        ord_vec
    }*/
    #[allow(non_snake_case)]
    fn select_random_individuals_from_archive(
        &self,
        size_selection: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let mut rng = rand::rng();
        let selected= self
            .archive_map
            .saved_point
            .choose_multiple(&mut rng, size_selection);
        let mut chossen = Vec::with_capacity(size_selection);
        for el in selected {
            println!("{:?}", *el);
            chossen.push(el.genotype.clone());
        }
        Ok(chossen)
    }
    fn store_archive_in_csv(&self, action_mode : String ) {
        match File::create(self.csv_archive_name.clone()) {
            Ok(mut f) => {
                for pt in self.archive_map.saved_point.iter() {
                    let _ = write!(f,"[[");
                    for (i, value) in pt.genotype.iter().enumerate() {
                        if i != 0 {
                            let _ = write!(f, ",");
                        }
                        let _ = write!(f, "{}", value);
                    }
                    
                    write!(f, "],{}", pt.fitness.to_string()).unwrap();
                    write!(f, ",{}",action_mode).unwrap();
                }
                let _ = writeln!(f);
                println!("archive stored");
            },Err(e) => {
                eprintln!("{}",e);
                println!("archive not stored");
            }
        }
        
    }
    fn store_several_element_in_archive(&mut self, list: &Bound<'_, PyList>) {
        println!("Store a lot a element");
        for element in list {
            let _  = self.store_one_new_element_in_archive(element.extract().unwrap());
        }
    }
    fn set_max_depth(&mut self, depth : u32) {
        self.archive_map.max_depth = depth;
    }
    fn set_max_precision(&mut self, precision : Vec<f64>) {
        self.archive_map.max_precision = precision;
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn dynamic_map_qd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Archive>()?;
    Ok(())
}

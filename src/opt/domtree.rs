use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ir::{Block, BlockId, FunctionDefinition};

pub trait DomTree {
    fn init_block(&self) -> BlockId;
    fn blocks(&self) -> &BTreeMap<BlockId, Block>;

    // https://eli.thegreenplace.net/2015/directed-graph-traversal-orderings-and-applications-to-data-flow-analysis/
    fn reverse_post_order(&self) -> Vec<BlockId> {
        let mut visited: HashSet<BlockId> = HashSet::new();
        let mut order: Vec<BlockId> = vec![];
        self.dfs_walker(self.init_block(), &mut visited, &mut order);
        order.reverse();
        assert_eq!(order[0], self.init_block());
        order
    }

    fn calculate_pred(&self) -> HashMap<BlockId, HashSet<BlockId>> {
        calculate_pred_inner(self.init_block(), self.blocks())
    }

    fn calculate_idom(
        &self,
        dom: &HashMap<BlockId, HashSet<BlockId>>,
        rpo: &[BlockId],
    ) -> HashMap<BlockId, BlockId> {
        let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
        'outer: for b in rpo.iter().filter(|&&x| x != self.init_block()) {
            for a in rpo.iter() {
                if dom.get(b).unwrap()
                    == &dom
                        .get(a)
                        .unwrap()
                        .union(&HashSet::from([*b]))
                        .cloned()
                        .collect::<HashSet<BlockId>>()
                {
                    let None = idom.insert(*b, *a) else {
                        unreachable!()
                    };
                    continue 'outer;
                }
            }
            unreachable!()
        }
        idom
    }

    // https://en.wikipedia.org/wiki/Dominator_(graph_theory)
    // A -> [A, B, C] means B dominate A, C dominate A
    // if a block is unreachable(not block_init, and no pred): not contained in return value
    fn calculate_dominator(
        &self,
        pred: &HashMap<BlockId, HashSet<BlockId>>,
    ) -> HashMap<BlockId, HashSet<BlockId>> {
        #[allow(non_snake_case)]
        let N: HashSet<BlockId> = self.blocks().iter().map(|(&bid, _)| bid).collect();

        // TODO: in another order
        let iter = self.blocks().iter().filter_map(|(bid, _)| {
            if bid != &self.init_block() {
                pred.get(bid).map(|pred| (bid, pred))
            } else {
                None
            }
        });

        let mut res: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
        let None = res.insert(self.init_block(), HashSet::from([self.init_block()])) else {
            unreachable!()
        };
        for (bid, _) in iter.clone() {
            let None = res.insert(*bid, N.clone()) else {
                unreachable!()
            };
        }

        let mut changed = true;
        while changed {
            changed = false;
            for (bid, pred) in iter.clone() {
                let x: HashSet<BlockId> = pred
                    .iter()
                    .filter_map(|x| res.get(x))
                    .fold(N.clone(), |a, b| a.intersection(b).cloned().collect())
                    .union(&HashSet::from([*bid]))
                    .cloned()
                    .collect();

                if &x != res.get(bid).unwrap() {
                    changed = true;
                    let Some(_) = res.insert(*bid, x) else {
                        unreachable!()
                    };
                }
            }
        }
        res
    }

    fn dfs_walker(&self, node: BlockId, visited: &mut HashSet<BlockId>, order: &mut Vec<BlockId>) {
        let true = visited.insert(node) else {
            unreachable!()
        };
        for succ in self.blocks().get(&node).unwrap().exit.walk_jump_bid() {
            if !visited.contains(&succ) {
                self.dfs_walker(succ, visited, order);
            }
        }
        order.push(node);
    }
}

pub fn calculate_pred_inner(
    bid_init: BlockId,
    blocks: &BTreeMap<BlockId, Block>,
) -> HashMap<BlockId, HashSet<BlockId>> {
    let mut hm: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
    for (&id, b) in blocks {
        for next_id in b.exit.walk_jump_bid() {
            let _b = hm.entry(next_id).or_default().insert(id);
        }
    }
    let None = hm.insert(bid_init, HashSet::new()) else {
        unreachable!()
    };
    hm
}

impl DomTree for FunctionDefinition {
    fn init_block(&self) -> BlockId {
        self.bid_init
    }

    fn blocks(&self) -> &BTreeMap<BlockId, Block> {
        &self.blocks
    }
}

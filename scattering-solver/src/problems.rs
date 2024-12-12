mod single_channel;
use quantum::problems_impl;
use self::single_channel::SingleChannel;

pub mod multi_channel;
use multi_channel::MultiChannelProblems;


pub struct Problems;

problems_impl!(Problems, "test",
    "single channel" => SingleChannel::select,
    "multi channel" => MultiChannelProblems::select
);
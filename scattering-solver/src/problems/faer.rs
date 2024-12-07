mod two_channel;
mod many_channels;

use many_channels::ManyChannels;
use quantum::problems_impl;
use two_channel::TwoChannel;

pub struct FaerProblems;

problems_impl!(FaerProblems, "faer",
    "two channel" => TwoChannel::select,
    "many channels" => ManyChannels::select
);
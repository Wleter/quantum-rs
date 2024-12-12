mod two_channel;
mod many_channels;

use many_channels::ManyChannels;
use quantum::problems_impl;
use two_channel::TwoChannel;

pub struct MultiChannelProblems;

problems_impl!(MultiChannelProblems, "multi channel",
    "two channel" => TwoChannel::select,
    "many channels" => ManyChannels::select
);
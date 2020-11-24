import { ITweetsModel } from '../../interfaces/tweets.interfaces';
import { IUserModel } from '../../interfaces/users.interfaces';
import { InjectModel } from '@nestjs/mongoose';
import { Injectable } from '@nestjs/common';
import { Model } from 'mongoose';

@Injectable()
export class RegionsService {
  constructor(
    @InjectModel('tweets') private readonly tweetModel: Model<ITweetsModel>,
    @InjectModel('users') private readonly userModel: Model<IUserModel>,
  ) {}

  getCommonRegions = async () =>
    await this.tweetModel.aggregate([
      {
        $match: {
          'location.city': {
            $exists: true,
          },
          'accuracy.political': {
            $gte: 0.65,
          },
          'sentimentScore.predominant': 'NEGATIVE',
        },
      },
      {
        $group: {
          _id: '$location.city',
          counter: {
            $sum: 1,
          },
        },
      },
      {
        $sort: {
          counter: -1,
        },
      },
      {
        $limit: 25,
      },
    ]);

  autoCompleteRegion = async (name: string) =>
    await this.tweetModel.aggregate([
      {
        $match: {
          $and: [
            { 'location.city': new RegExp(`.*${name}.*`, 'i') },
            { 'location.city': { $exists: true } },
          ],
        },
      },
      {
        $group: {
          _id: '$location.city',
        },
      },
      {
        $sort: {
          _id: 1,
        },
      },
    ]);

  getRegionDetailsByName = async (name: string) => {
    const political = await this.tweetModel
      .find({
        'accuracy.political': {
          $gte: 0.65,
        },
        'location.city': name,
      })
      .countDocuments();
    const hate = await this.tweetModel
      .find({
        'accuracy.political': {
          $gte: 0.65,
        },
        'sentimentScore.predominant': 'NEGATIVE',
        'location.city': name,
      })
      .countDocuments();
    const randomTweets = await this.tweetModel
      .find({
        'accuracy.political': {
          $gte: 0.65,
        },
        'sentimentScore.predominant': 'NEGATIVE',
        'location.city': name,
      })
      .limit(3)
      .select('tweetId text')
      .lean();
    return { political, hate, randomTweets };
  };

  getRegionDetailsByLatLon = async (lat: number, long: number) => {
    const political = await this.tweetModel
      .find({
        'accuracy.political': {
          $gte: 0.65,
        },
        'location.latitude': lat,
        'location.longitude': long,
      })
      .countDocuments();
    const hate = await this.tweetModel
      .find({
        'accuracy.political': {
          $gte: 0.65,
        },
        'sentimentScore.predominant': 'NEGATIVE',
        'location.latitude': lat,
        'location.longitude': long,
      })
      .countDocuments();
    const randomTweets = await this.tweetModel
      .find({
        'accuracy.political': {
          $gte: 0.65,
        },
        'sentimentScore.predominant': 'NEGATIVE',
        'location.latitude': lat,
        'location.longitude': long,
      })
      .limit(3)
      .select('tweetId text')
      .lean();
    return { political, hate, randomTweets };
  };
}

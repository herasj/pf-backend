import {
  ITweetResponse,
  ITweetsModel,
} from '../../interfaces/tweets.interfaces';
import { IUserModel } from '../../interfaces/users.interfaces';
import { UpdatePoliticalTweetDTO } from './dtos/tweet.dto';
import { Injectable, Inject } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';

@Injectable()
export class TweetsService {
  constructor(
    @InjectModel('tweets') private readonly tweetModel: Model<ITweetsModel>,
    @InjectModel('users') private readonly userModel: Model<IUserModel>,
  ) {}

  findRandom = async (): Promise<ITweetsModel[]> =>
    await this.tweetModel.aggregate([
      { $match: { political: { $exists: false } } },
      { $sample: { size: 1 } },
    ]);

    findRandomPolitical = async (): Promise<ITweetsModel[]> =>
    await this.tweetModel.aggregate([
      { $match: { political: { $exists: true } } },
      { $sample: { size: 1 } },
    ]);

  getDetails = async (tweetId: string) =>{
    const tweet = await this.tweetModel.findOne({ tweetId }).lean();
    const user: any = await this.userModel.findOne({userId: tweet.userId}).select('userId name username verified counter' ).lean()
    const political = await this.tweetModel
      .find({ userId: user.userId, political: true })
      .countDocuments();
    const hate = await this.tweetModel
      .find({
        userId: user.userId,
        political: true,
        'sentimentScore.predominant': 'NEGATIVE',
      })
      .countDocuments();
    user.counter = { political, hate };
    return {tweet, user}
  }
    
  getTodayTweets = async (): Promise<number> =>
    await this.tweetModel
      .find({ 'sentimentScore.createdAt': { $gte: new Date() } })
      .countDocuments()
      .lean();

  getUserTweets = async (
    userId: string,
    page: number,
  ): Promise<ITweetResponse[]> =>
    await this.tweetModel.aggregate([
      {
        $match: {
          political: true,
          'sentimentScore.predominant': 'NEGATIVE',
          userId,
        },
      },
      {
        $lookup: {
          from: 'users',
          localField: 'userId',
          foreignField: 'userId',
          as: 'user',
        },
      },
      {
        $unwind: {
          path: '$user',
        },
      },
      {
        $project: {
          tweetId: 1,
          username: '$user.username',
          name: '$user.name',
          createdAt: {
            $toDate: '$createdAt',
          },
        },
      },
      {
        $sort: {
          createdAt: -1,
        },
      },
      {
        $skip: 10 * (page - 1),
      },
      {
        $limit: 10,
      },
    ]);

  getLatest = async (page: number): Promise<ITweetResponse[]> =>
    await this.tweetModel.aggregate([
      {
        $match: {
          political: true,
          'sentimentScore.predominant': 'NEGATIVE',
        },
      },
      {
        $lookup: {
          from: 'users',
          localField: 'userId',
          foreignField: 'userId',
          as: 'user',
        },
      },
      {
        $unwind: {
          path: '$user',
        },
      },
      {
        $project: {
          tweetId: 1,
          sentimentScore: 1,
          username: '$user.username',
          name: '$user.name',
          createdAt: {
            $toDate: '$createdAt',
          },
        },
      },
      {
        $sort: {
          createdAt: -1,
        },
      },
      {
        $skip: 10 * (page - 1),
      },
      {
        $limit: 10,
      },
    ]);

  getNegativeTweets = async (): Promise<ITweetsModel[]> =>
    await this.tweetModel
      .find({ 'sentimentScore.predominant': 'NEGATIVE' })
      .lean();

  countTweets = async (): Promise<number> =>
    await this.tweetModel
      .find({ political: { $exists: true } })
      .countDocuments();

  countPoliticalTweets = async (): Promise<number> =>
    await this.tweetModel.find({ political: true }).countDocuments();

  getLocation = async (): Promise<ITweetsModel[]> =>
    await this.tweetModel.aggregate([
      {
        $match: {
          political: true,
        },
      },
      {
        $group: {
          _id: '$location',
          count: {
            $sum: 1,
          },
        },
      },
      {
        $project: {
          lcoation: '$_id',
          count: 1,
          _id: 0,
        },
      },
      {
        $sort: {
          count: -1,
        },
      },
    ]);

  updateOne = async (data: UpdatePoliticalTweetDTO) =>
    await this.tweetModel
      .findByIdAndUpdate(data._id, {
        political: data.political,
      })
      .lean();
}

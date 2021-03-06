import {
  ITweetResponse,
  ITweetsModel,
} from '../../interfaces/tweets.interfaces';
import { IUserModel } from '../../interfaces/users.interfaces';
import { UpdatePoliticalTweetDTO } from './dtos/tweet.dto';
import { Injectable, Inject } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import * as moment from 'moment';
import { Model } from 'mongoose';

@Injectable()
export class TweetsService {
  private regex = new RegExp(
    'izquierda|derecha|tibio|Petro|Uribe|Duque|Farc|ELN|Disidencias|castrochavismo|mermelada|corruptos|petristas|uribestia',
    'ig',
  );
  constructor(
    @InjectModel('tweets') private readonly tweetModel: Model<ITweetsModel>,
    @InjectModel('users') private readonly userModel: Model<IUserModel>,
  ) {}

  findRandom = async (): Promise<ITweetsModel[]> =>
    await this.tweetModel.aggregate([
      {
        $match: {
          $or: [{ political: true }, { 'accuracy.political': { $gte: 0.6 } }],
          polarization: {$exists: false}
        },
      },
      { $sample: { size: 1 } },
    ]);

  findRandomPolitical = async (): Promise<ITweetsModel[]> =>
    await this.tweetModel.aggregate([
      { $match: { political: { $exists: true } } },
      { $sample: { size: 1 } },
    ]);

  graphInfo = async () =>
    await this.tweetModel.aggregate([
      {
        $match: {
          $or: [
            {
              political: true,
            },
            {
              'accuracy.political': {
                $gte: 0.6,
              },
            },
          ],
          'sentimentScore.predominant': 'NEGATIVE',
        },
      },
      {
        $project: {
          createdAt: {
            $dateToString: {
              date: '$sentimentScore.createdAt',
              format: '%Y-%m-%d',
            },
          },
        },
      },
      {
        $group: {
          _id: '$createdAt',
          counter: {
            $sum: 1,
          },
        },
      },
      {
        $sort: {
          _id: -1,
        },
      },
      {
        $limit: 15,
      },
      {
        $sort: {
          _id: 1,
        },
      },
    ]);

  getDetails = async (tweetId: string) => {
    const tweet = await this.tweetModel.findOne({ tweetId }).lean();
    const user: any = await this.userModel
      .findOne({ userId: tweet.userId })
      .select('userId name username verified counter')
      .lean();
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
    return { tweet, user };
  };

  getTodayTweets = async (): Promise<number> =>
    await this.tweetModel
      .find({
        'sentimentScore.createdAt': {
          $gte: moment()
            .startOf('day')
            .toDate(),
        },
      })
      .countDocuments()
      .lean();

  getUserTweets = async (
    userId: string,
    page: number,
  ): Promise<ITweetResponse[]> =>
    await this.tweetModel.aggregate([
      {
        $match: {
          $or: [
            {
              political: true,
            },
            {
              'accuracy.political': {
                $gte: 0.6,
              },
            },
          ],
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
          sentimentScore: 1,
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
          $or: [
            {
              political: true,
            },
            {
              'accuracy.political': {
                $gte: 0.6,
              },
            },
          ],
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
      .find({})
      .or([
        { political: { $exists: true } },
        {
          'accuracy.political': { $exists: true },
        },
      ])
      .countDocuments();

  countPoliticalTweets = async (): Promise<number> =>
    await this.tweetModel
      .find()
      .or([
        { political: true },
        {
          'accuracy.political': {
            $gte: 0.6,
          },
        },
      ])
      .countDocuments();

  getLocation = async (): Promise<ITweetsModel[]> =>
    await this.tweetModel.aggregate([
      {
        $match: {
          $or: [
            {
              political: true,
            },
            {
              'accuracy.political': {
                $gte: 0.6,
              },
            },
          ],
          'sentimentScore.predominant': 'NEGATIVE',
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

  updateOne = async (_id: string, polarization: boolean) =>
    await this.tweetModel
      .findByIdAndUpdate(_id, {
        polarization,
      })
      .lean();
}

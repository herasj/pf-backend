import { ITweetsModel } from '../../interfaces/tweets.interfaces';
import { UpdatePoliticalTweetDTO } from './dtos/tweet.dto';
import { Injectable, Inject } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';

@Injectable()
export class TweetsService {
  constructor(
    @InjectModel('tweets') private readonly tweetModel: Model<ITweetsModel>,
  ) {}
  findRandom = async (): Promise<ITweetsModel[]> =>
    await this.tweetModel.aggregate([
      { $match: { political: { $exists: false } } },
      { $sample: { size: 1 } },
    ]);
  updateOne = async (data: UpdatePoliticalTweetDTO) =>
    await this.tweetModel
      .findByIdAndUpdate(data._id, {
        political: data.political,
      })
      .lean();
}

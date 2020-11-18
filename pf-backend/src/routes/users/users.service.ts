import { ITweetsModel } from '../../interfaces/tweets.interfaces';
import { IUserModel } from '../../interfaces/users.interfaces';
import { InjectModel } from '@nestjs/mongoose';
import { Injectable } from '@nestjs/common';
import { Model } from 'mongoose';

@Injectable()
export class UsersService {
  constructor(
    @InjectModel('tweets') private readonly tweetModel: Model<ITweetsModel>,
    @InjectModel('users') private readonly userModel: Model<IUserModel>,
  ) {}

  getCommonUsers = async () =>
    await this.userModel
      .find({})
      .limit(25)
      .select('userId name username counter')
      .sort('counter')
      .lean();

  getUserDetails = async (_id: string) => {
    const user = await this.userModel.findById(_id).lean();
  };

  getUserDetailsByUsername = async (username: string) => {
    const user: any = await this.userModel.findOne({ username }).lean();
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
    return user;
  };
}

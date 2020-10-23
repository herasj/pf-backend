import { IUserModel } from '../../interfaces/users.interfaces';
import { InjectModel } from '@nestjs/mongoose';
import { Injectable } from '@nestjs/common';
import { Model } from 'mongoose';

@Injectable()
export class UsersService {
  constructor(
    @InjectModel('users') private readonly userModel: Model<IUserModel>,
  ) {}

  getCommonUsers = async () =>
    await this.userModel
      .find({})
      .limit(25)
      .select('name username counter')
      .sort('counter')
      .lean();

  getUserDetails = async (_id: string) =>
    await this.userModel.findById(_id).lean();
}
